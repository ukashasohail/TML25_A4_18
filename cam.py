import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import (
    GradCAM, FEM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise, KPCA_CAM, ShapleyCAM,
    FinerCAM
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputReST

def compute_iou_from_cam(cam, threshold=0.5, ground_truth_bbox=None, ground_truth_mask=None):
    """
    Compute Intersection over Union (IoU) between CAM and ground truth.
    
    Args:
        cam: 2D numpy array representing the CAM
        threshold: float, threshold to binarize the CAM (default: 0.5)
        ground_truth_bbox: tuple (x1, y1, x2, y2) representing bounding box coordinates
        ground_truth_mask: 2D numpy array representing ground truth binary mask
    
    Returns:
        iou_score: float, IoU score between 0 and 1
        cam_bbox: tuple, bounding box extracted from CAM
        binary_cam: 2D numpy array, thresholded CAM
    """
    # Normalize CAM to [0, 1] range
    cam_normalized = (cam - cam.min()) / (cam.max() - cam.min())
    
    # Threshold CAM to create binary mask
    binary_cam = (cam_normalized > threshold).astype(np.uint8)
    
    # Extract bounding box from CAM using connected components
    contours, _ = cv2.findContours(binary_cam, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return 0.0, None, binary_cam
    
    # Find largest contour (assuming it's the main object)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cam_bbox = (x, y, x + w, y + h)
    
    if ground_truth_bbox is not None:
        # Compute IoU with ground truth bounding box
        iou_score = compute_bbox_iou(cam_bbox, ground_truth_bbox)
    elif ground_truth_mask is not None:
        # Compute IoU with ground truth mask
        iou_score = compute_mask_iou(binary_cam, ground_truth_mask)
    else:
        print("Warning: No ground truth provided. Returning CAM bbox only.")
        return None, cam_bbox, binary_cam
    
    return iou_score, cam_bbox, binary_cam


def compute_bbox_iou(bbox1, bbox2):
    """
    Compute IoU between two bounding boxes.
    
    Args:
        bbox1: tuple (x1, y1, x2, y2)
        bbox2: tuple (x1, y1, x2, y2)
    
    Returns:
        iou: float, IoU score
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def compute_mask_iou(mask1, mask2):
    """
    Compute IoU between two binary masks.
    
    Args:
        mask1: 2D numpy array (binary)
        mask2: 2D numpy array (binary)
    
    Returns:
        iou: float, IoU score
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    return intersection / union if union > 0 else 0.0


def visualize_iou_result(rgb_img, cam, binary_cam, cam_bbox, ground_truth_bbox=None, iou_score=None):
    """
    Create visualization showing CAM, binary mask, and bounding boxes.
    
    Returns:
        visualization: numpy array containing the combined visualization
    """
    # Create CAM overlay
    cam_overlay = show_cam_on_image(rgb_img, cam, use_rgb=True)
    
    # Create binary mask visualization
    binary_vis = np.stack([binary_cam * 255] * 3, axis=-1).astype(np.uint8)
    
    # Draw bounding boxes
    bbox_vis = (rgb_img * 255).astype(np.uint8).copy()
    
    if cam_bbox is not None:
        # Draw CAM bounding box in green
        cv2.rectangle(bbox_vis, (cam_bbox[0], cam_bbox[1]), (cam_bbox[2], cam_bbox[3]), (0, 255, 0), 2)
        cv2.putText(bbox_vis, 'CAM', (cam_bbox[0], cam_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    if ground_truth_bbox is not None:
        # Draw ground truth bounding box in red
        cv2.rectangle(bbox_vis, (ground_truth_bbox[0], ground_truth_bbox[1]), 
                     (ground_truth_bbox[2], ground_truth_bbox[3]), (255, 0, 0), 2)
        cv2.putText(bbox_vis, 'GT', (ground_truth_bbox[0], ground_truth_bbox[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    if iou_score is not None:
        cv2.putText(bbox_vis, f'IoU: {iou_score:.3f}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Combine visualizations
    combined = np.hstack([
        (rgb_img * 255).astype(np.uint8),
        cam_overlay,
        binary_vis,
        bbox_vis
    ])
    
    return combined


image_paths = [
    "n01443537_goldfish",
    "n01491361_tiger_shark",
    "n01608432_kite",
    "n01616318_vulture",
    "n01677366_common_iguana",
    "n02007558_flamingo",
    "n02018207_American_coot",
    "n02098286_West_Highland_white_terrier",
    "n04037443_racer",
    "n07747607_orange"
]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='mps',
                        help='Torch device to use')
    parser.add_argument(
        '--image-path',
        type=str,
        default='../imagenet-sample-images/n07747607_orange.JPEG',
        help='Input image path')
    parser.add_argument('--aug-smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen-smooth',
        action='store_true',
        help='Reduce noise by taking the first principle component'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=[
                            'gradcam', 'fem', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise', 'kpcacam', 'shapleycam',
                            'finercam'
                        ],
                        help='CAM method')

    parser.add_argument('--output-dir', type=str, default='./gradcam_results/',
                        help='Output directory to save the images')
    
    # Model-specific arguments
    parser.add_argument('--model-type', type=str, default='resnet50',
                        choices=['resnet50', 'cnn'],
                        help='Model type to use')
    parser.add_argument('--model-path', type=str, default='./deep_fake_project/cnn_best.pt',
                        help='Path to saved model state dict')
    parser.add_argument('--cnn-in-features', type=int, default=1,
                        help='Input features for CNN model')
    parser.add_argument('--cnn-out-dim', type=int, default=2,
                        help='Output dimensions for CNN model')
    
    # IoU-related arguments
    parser.add_argument('--compute-iou', action='store_true',
                        help='Compute IoU metric')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='Threshold for binarizing CAM for IoU computation')
    parser.add_argument('--ground-truth-bbox', type=str, default=None,
                        help='Ground truth bounding box as "x1,y1,x2,y2"')
    
    args = parser.parse_args()
    
    if args.device:
        print(f'Using device "{args.device}" for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "fem": FEM,
        "gradcamelementwise": GradCAMElementWise,
        'kpcacam': KPCA_CAM,
        'shapleycam': ShapleyCAM,
        'finercam': FinerCAM
    }

    if args.device=='hpu':
        import habana_frameworks.torch.core as htcore
        
    device = torch.device(args.device)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device).eval()

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    
    target_layers = [model.layer4]

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]).to(args.device)

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [ClassifierOutputTarget(281)]
    # targets = [ClassifierOutputReST(281)]
    targets = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[args.method]
    with cam_algorithm(model=model,
                       target_layers=target_layers) as cam:

        cam.batch_size = 32  # Reduced batch size for custom models
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)

        grayscale_cam = grayscale_cam[0, :]

        # Resize CAM to match RGB image dimensions for visualization
        if grayscale_cam.shape != rgb_img.shape[:2]:
            grayscale_cam = cv2.resize(grayscale_cam, (rgb_img.shape[1], rgb_img.shape[0]))

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    # Guided Backpropagation
    try:
        gb_model = GuidedBackpropReLUModel(model=model, device=args.device)
        gb = gb_model(input_tensor, target_category=None)

        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)
    except Exception as e:
        print(f"Warning: Guided Backpropagation failed: {e}")
        gb = np.zeros_like((rgb_img * 255).astype(np.uint8))
        cam_gb = np.zeros_like((rgb_img * 255).astype(np.uint8))

    # Compute IoU if requested
    if args.compute_iou:
        ground_truth_bbox = None
        if args.ground_truth_bbox:
            try:
                bbox_coords = [int(x) for x in args.ground_truth_bbox.split(',')]
                if len(bbox_coords) == 4:
                    ground_truth_bbox = tuple(bbox_coords)
                else:
                    print("Error: Ground truth bounding box must have 4 coordinates (x1,y1,x2,y2).")
            except ValueError:
                print("Error: Invalid format for ground truth bounding box. Use x1,y1,x2,y2.")
        
        iou_score, cam_bbox, binary_cam = compute_iou_from_cam(
            grayscale_cam, 
            threshold=args.iou_threshold,
            ground_truth_bbox=ground_truth_bbox
        )
        
        if iou_score is not None:
            print(f"IoU Score: {iou_score:.4f}")
            print(f"CAM Bounding Box: {cam_bbox}")
            if ground_truth_bbox:
                print(f"Ground Truth Bounding Box: {ground_truth_bbox}")
        
        # Create IoU visualization
        iou_visualization = visualize_iou_result(
            rgb_img, grayscale_cam, binary_cam, cam_bbox, 
            ground_truth_bbox, iou_score
        )
        
        # Save IoU visualization
        os.makedirs(args.output_dir, exist_ok=True)
        iou_output_path = os.path.join(args.output_dir, f'{args.method}_iou_analysis.jpg')
        cv2.imwrite(iou_output_path, cv2.cvtColor(iou_visualization, cv2.COLOR_RGB2BGR))
        print(f"IoU analysis saved to: {iou_output_path}")

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    cam_output_path = os.path.join(args.output_dir, f'{args.method}_cam.jpg')
    gb_output_path = os.path.join(args.output_dir, f'{args.method}_gb.jpg')
    cam_gb_output_path = os.path.join(args.output_dir, f'{args.method}_cam_gb.jpg')

    cv2.imwrite(cam_output_path, cam_image)
    cv2.imwrite(gb_output_path, gb)
    cv2.imwrite(cam_gb_output_path, cam_gb)