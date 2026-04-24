import os
from glob import glob

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.ops as ops
from PIL import Image
from transformers import (
    AutoProcessor,
    GroundingDinoForObjectDetection,
    SamModel,
    SamProcessor,
)


class FrameProcessor:
    def __init__(self, video_dir, max_frames=1000, skip_frames=50, batch_size=4):
        self.video_dir = video_dir
        self.max_frames = max_frames
        self.skip_frames = skip_frames
        self.batch_size = batch_size
        self.video_files = self._get_video_files()

    def _get_video_files(self):
        """
        Get all video files from directory, sorted by number
        """
        video_files = []
        for file in sorted(os.listdir(self.video_dir)):
            if file.endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_files.append(os.path.join(self.video_dir, file))

        print(f"Found {len(video_files)} video files:")
        for vf in video_files:
            print(f"  - {os.path.basename(vf)}")

        video_files.sort(
            key=lambda x: int("".join(filter(str.isdigit, os.path.basename(x))))
        )

        return video_files

    def get_frame_batch(self):
        """
        Generator that yields frame batches from multiple videos
        """
        frame_batch = []
        frame_indices = []
        count = 0
        global_frame_idx = 0

        for video_path in self.video_files:
            if count >= self.max_frames:
                break

            print(f"\nProcessing video: {os.path.basename(video_path)}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open {video_path}")
                continue

            frame_number = 0

            while cap.isOpened() and count < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_number % self.skip_frames == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_batch.append(frame_rgb)
                    frame_indices.append((os.path.basename(video_path), frame_number))
                    count += 1
                    global_frame_idx += 1

                    # Yield batch when full
                    if len(frame_batch) == self.batch_size:
                        yield frame_batch, frame_indices
                        frame_batch = []
                        frame_indices = []

                frame_number += 1

            cap.release()

        # Yield remaining frames
        if frame_batch:
            yield frame_batch, frame_indices


class ImageListProcessor:
    def __init__(self, image_list_file, max_frames=1000, batch_size=4):
        self.image_list_file = image_list_file
        self.max_frames = max_frames
        self.batch_size = batch_size
        self.image_paths = []
        self._load_image_list()

    def _load_image_list(self):
        """
        Load image paths from text file (one path per line)
        """
        self.image_paths = glob(
            os.path.join(self.image_list_file, "*.jpg")
        )  # Adjust extension if needed

        # Limit to max_frames
        self.image_paths = self.image_paths[: self.max_frames]
        self.image_paths.sort(
            key=lambda x: int(
                "".join(filter(str.isdigit, os.path.basename(x).split("_")[1]))
            )
        )

        print(f"Loaded {len(self.image_paths)} image paths from {self.image_list_file}")

    def get_frame_batch(self):
        """
        Generator that yields image batches without loading all images at once
        """
        frame_batch = []
        frame_indices = []

        for idx, image_path in enumerate(self.image_paths):
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not read image {image_path}")
                    continue

                frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frame_batch.append(frame_rgb)
                frame_indices.append(image_path)

                # Yield batch when full
                if len(frame_batch) == self.batch_size:
                    yield frame_batch, frame_indices
                    frame_batch = []
                    frame_indices = []
            except Exception as e:
                print(f"Error reading {image_path}: {e}")
                continue

        # Yield remaining frames
        if frame_batch:
            yield frame_batch, frame_indices


def process_batch_with_dino(
    frames_batch, processor, model, device, text_prompt="object"
):
    """
    Process a single batch of frames
    """
    results = []
    input_ids_batch = []

    for frame in frames_batch:
        inputs = processor(images=frame, text=text_prompt, return_tensors="pt").to(
            device
        )

        with torch.no_grad():
            outputs = model(**inputs)

        results.append(outputs)
        input_ids_batch.append(inputs.input_ids)

    return results, input_ids_batch


def post_process_batch(
    processor, outputs_list, input_ids_batch, frames_batch, threshold=0.2
):
    """
    Post-process a batch of detections
    """
    processed_results = []

    for idx, (outputs, input_ids) in enumerate(zip(outputs_list, input_ids_batch)):
        h, w = frames_batch[idx].shape[:2]
        target_sizes = [(h, w)]

        result = processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=input_ids,
            threshold=threshold,
            target_sizes=target_sizes,
        )

        processed_results.extend(result)

    return processed_results


def apply_nms(result, nms_threshold=0.3):
    boxes = (
        result["boxes"].cpu().numpy()
        if torch.is_tensor(result["boxes"])
        else np.array(result["boxes"])
    )
    scores = (
        result["scores"].cpu().numpy()
        if torch.is_tensor(result["scores"])
        else np.array(result["scores"])
    )
    text_labels = result.get("text_labels", [])

    if len(boxes) == 0:
        return result

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)

    keep_indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold=nms_threshold)

    filtered_boxes = boxes[keep_indices.cpu().numpy()]
    filtered_scores = scores[keep_indices.cpu().numpy()]
    filtered_labels = (
        [text_labels[i] for i in keep_indices.cpu().numpy()] if text_labels else []
    )

    result["boxes"] = torch.tensor(filtered_boxes)
    result["scores"] = torch.tensor(filtered_scores)
    result["text_labels"] = filtered_labels

    return result


def filter_large_boxes(result, frame_shape, max_coverage=0.7):
    h, w = frame_shape[:2]
    image_area = h * w

    boxes = (
        result["boxes"].cpu().numpy()
        if torch.is_tensor(result["boxes"])
        else np.array(result["boxes"])
    )
    scores = (
        result["scores"].cpu().numpy()
        if torch.is_tensor(result["scores"])
        else np.array(result["scores"])
    )
    text_labels = result.get("text_labels", [])

    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []

    for box_idx, (box, score) in enumerate(zip(boxes, scores)):
        x_min, y_min, x_max, y_max = box

        x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
        x_max, y_max = min(w, int(x_max)), min(h, int(y_max))

        width = x_max - x_min
        height = y_max - y_min
        box_area = width * height
        coverage = box_area / image_area

        if coverage > max_coverage:
            continue

        if width < 5 or height < 5:
            continue

        filtered_boxes.append([x_min, y_min, x_max, y_max])
        filtered_scores.append(score)
        if text_labels and box_idx < len(text_labels):
            filtered_labels.append(text_labels[box_idx])

    result["boxes"] = (
        torch.tensor(filtered_boxes) if filtered_boxes else torch.tensor([])
    )
    result["scores"] = (
        torch.tensor(filtered_scores) if filtered_scores else torch.tensor([])
    )
    result["text_labels"] = filtered_labels

    return result


def segment_with_sam(frame, boxes, sam_processor, sam_model, device):
    """
    Use SAM to segment objects within detected boxes
    """
    if len(boxes) == 0:
        return []

    frame_pil = Image.fromarray(frame.astype("uint8"))

    if torch.is_tensor(boxes):
        boxes = boxes.cpu().numpy().tolist()
    else:
        boxes = boxes.tolist() if isinstance(boxes, np.ndarray) else boxes

    input_boxes = [boxes]

    inputs = sam_processor(frame_pil, input_boxes=input_boxes, return_tensors="pt").to(
        device
    )

    with torch.no_grad():
        outputs = sam_model(**inputs)

    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
    )

    return masks[0].cpu().numpy()


def create_output_directory(base_dir, dataset_name):
    """
    Create output directory with dataset name and timestamp
    """
    output_dir = os.path.join(base_dir, dataset_name)
    for option in ["mask", "foreground"]:
        output_subdir = os.path.join(output_dir, option)
        os.makedirs(output_subdir, exist_ok=True)

    return output_dir


def save_segmentation_results(
    frame,
    masks,
    scores,
    text_labels,
    frame_idx,
    original_frame_num,
    output_dir,
    dataset_type,
):
    """
    Save outputs per frame: mask, foreground, and background

    Args:
        output_type: "mask", "foreground", and "background"
    """

    filename = f"frame_{frame_idx:05d}.png"

    # Combine all masks into single binary image
    combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    for mask in masks:
        combined_mask = np.logical_or(combined_mask, mask[0]).astype(np.uint8) * 255
    mask_output_path = os.path.join(output_dir, "mask", filename)
    cv2.imwrite(mask_output_path, combined_mask)

    merge_combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=bool)
    for mask in masks:
        merge_combined_mask = np.logical_or(merge_combined_mask, mask[0])

    # foreground
    combined_foreground = np.zeros_like(frame)
    combined_foreground[merge_combined_mask] = frame[merge_combined_mask]
    combined_foreground_rgb = cv2.cvtColor(combined_foreground, cv2.COLOR_RGB2BGR)
    foreground_output_path = os.path.join(output_dir, "foreground", filename)
    cv2.imwrite(foreground_output_path, combined_foreground_rgb)

    # background
    combined_background = frame.copy()
    combined_background[merge_combined_mask] = 0
    combined_background_rgb = cv2.cvtColor(combined_background, cv2.COLOR_RGB2BGR)
    background_output_path = os.path.join(output_dir, filename)
    cv2.imwrite(background_output_path, combined_background_rgb)


def process_dataset(
    processor_generator,
    dataset_name,
    dataset_type,
    dino_processor,
    dino_model,
    sam_processor,
    sam_model,
    device,
    output_dir,
    batch_size=4,
    threshold=0.2,
    max_coverage=0.7,
    nms_threshold=0.3,
    visualize=False,
):
    """
    Process a single dataset (video or image list)

    Args:
        output_type: "mask", "foreground", and "background"
    """
    text_prompt = "grasper. bipolar. scissor. clipper. hook. irrigator. needle driver"

    print(f"\n{'='*80}")
    print(f"Processing {dataset_type.upper()}: {dataset_name}")
    print(f"{'='*80}")

    batch_count = 0
    global_frame_idx = 0

    for frames_batch, frame_indices in processor_generator.get_frame_batch():
        batch_count += 1
        print(f"\nBatch {batch_count} ({len(frames_batch)} frames)")

        # Step 1: Detection
        detection_outputs, input_ids_list = process_batch_with_dino(
            frames_batch, dino_processor, dino_model, device, text_prompt
        )

        # Step 2: Post-processing detections
        detection_results = post_process_batch(
            dino_processor, detection_outputs, input_ids_list, frames_batch, threshold
        )

        # Step 3: Segmentation
        for frame_offset, (frame, frame_num, result) in enumerate(
            zip(frames_batch, frame_indices, detection_results)
        ):
            frame_idx = global_frame_idx + frame_offset

            # Apply NMS
            result = apply_nms(result, nms_threshold=nms_threshold)

            # Filter large boxes
            result = filter_large_boxes(result, frame.shape, max_coverage=max_coverage)

            boxes = (
                result["boxes"].cpu().numpy()
                if torch.is_tensor(result["boxes"])
                else np.array(result["boxes"])
            )
            scores = (
                result["scores"].cpu().numpy()
                if torch.is_tensor(result["scores"])
                else np.array(result["scores"])
            )
            text_labels = result.get("text_labels", [])

            if len(boxes) > 0:
                masks = segment_with_sam(frame, boxes, sam_processor, sam_model, device)

                if dataset_type == "video":
                    print(
                        f"  Frame {frame_idx + 1} (video #{frame_num}): {len(boxes)} objects detected"
                    )
                else:
                    print(
                        f"  Frame {frame_idx + 1} ({os.path.basename(frame_num)}): {len(boxes)} objects detected"
                    )

                # Save results
                save_segmentation_results(
                    frame,
                    masks,
                    scores,
                    text_labels,
                    frame_idx,
                    frame_num,
                    output_dir,
                    dataset_type,
                )

                # Optional visualization
                if visualize and frame_idx % 10 == 0:
                    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
                    ax.imshow(frame)
                    for mask in masks:
                        ax.contour(mask[0], colors="red", linewidths=1)
                    ax.set_title(f"Frame {frame_idx + 1}")
                    ax.axis("off")
                    plt.tight_layout()
                    plt.show()
                    plt.pause(0.5)
                    plt.close(fig)
            else:
                if dataset_type == "video":
                    print(
                        f"  Frame {frame_idx + 1} (video #{frame_num}): No objects detected"
                    )
                else:
                    print(
                        f"  Frame {frame_idx + 1} ({os.path.basename(frame_num)}): No objects detected"
                    )

        global_frame_idx += len(frames_batch)

        # Clear memory
        del frames_batch, detection_outputs, detection_results
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"Dataset '{dataset_name}' processing complete!")
    return output_dir


if __name__ == "__main__":
    # Dataset configurations
    datasets = [
        {
            "type": "video",
            "name": "train/A",
            "path": "/mnt/sdc/cholec80/videos",
            "max_frames": 1500,
            "skip_frames": 25,
        },
        {
            "type": "image_list",
            "name": "train/B",
            "path": "/mnt/sdc/wdr/liver_repair/0-19",
            "max_frames": 2500,
        },
    ]

    batch_size = 4
    base_output_dir = "cholec2wdr"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("Loading GroundingDINO model...")
    dino_model_id = "IDEA-Research/grounding-dino-base"
    dino_processor = AutoProcessor.from_pretrained(dino_model_id)
    dino_model = GroundingDinoForObjectDetection.from_pretrained(dino_model_id).to(
        device
    )
    dino_model.eval()

    print("Loading SAM model...")
    sam_model_id = "facebook/sam-vit-base"
    sam_processor = SamProcessor.from_pretrained(sam_model_id)
    sam_model = SamModel.from_pretrained(sam_model_id).to(device)
    sam_model.eval()

    # Process each dataset
    all_output_dirs = {}

    for idx, dataset_config in enumerate(datasets):
        dataset_type = dataset_config["type"]
        dataset_name = dataset_config["name"]
        dataset_path = dataset_config["path"]
        max_frames = dataset_config["max_frames"]

        print(f"\n{'#'*80}")
        print(f"Initializing {dataset_type}: {dataset_name}")
        print(f"{'#'*80}")

        # Create output directory for this dataset
        output_dir = create_output_directory(base_output_dir, dataset_name)
        print(f"Output directory: {output_dir}\n")

        # Create appropriate processor
        if dataset_type == "video":
            skip_frames = dataset_config.get("skip_frames", 25)
            processor_generator = FrameProcessor(
                dataset_path, max_frames, skip_frames, batch_size
            )
        else:  # image_list
            processor_generator = ImageListProcessor(
                dataset_path, max_frames, batch_size
            )

        # Process dataset
        result_dir = process_dataset(
            processor_generator,
            dataset_name,
            dataset_type,
            dino_processor,
            dino_model,
            sam_processor,
            sam_model,
            device,
            output_dir,
            batch_size=batch_size,
            threshold=0.2,
            max_coverage=0.7,
            nms_threshold=0.3,
            visualize=False,
        )

        all_output_dirs[dataset_name] = result_dir

    # Summary
    print(f"\n{'='*80}")
    print("Processing Summary")
    print(f"{'='*80}")
    for dataset_name, output_dir in all_output_dirs.items():
        print(f"Dataset: {dataset_name}")
        print(f"  Output: {output_dir}")
    print(f"{'='*80}")
