import sys
sys.path.insert(0, "../../")
import torch
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from pc_processor.models.radiant3d import RADIANT3DNet
from pc_processor.models.pmf_net import PMFNet
from pc_processor.dataset.semantic_kitti import SemanticKitti
from pc_processor.dataset.perspective_view_loader import PerspectiveViewLoader
from pc_processor.metrics.iou_eval import IOUEval
from collections import Counter

pred_counter = Counter()

# ============================================================
# Configuration
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root = "/mnt/hdd_check/PMF/SemanticKITTI-FOV/dataset/sequences"

baseline_checkpoint = "/home/dat/lidar_research/experiments/PMF-SemanticKitti-15ep/log_SemanticKitti_PMFNet-resnet34_bs1-lr0.001_pmf15_controlled/checkpoint/best_IOU_model.pth"

checkpoint = "/home/dat/lidar_research/experiments/AdaptivePMF-V1-SemanticKitti/log_SemanticKitti_PMFNet-resnet34_bs1-lr0.0001_adaptive_pmf_v1_15ep/checkpoint/best_IOU_model.pth"

config_path = (
    "/home/dat/lidar_research/PMF/"
    "pc_processor/dataset/semantic_kitti/semantic-kitti.yaml"
)

feature_mean = torch.tensor(
    [12.12, 10.88, 0.23, -1.04, 0.21],
    dtype=torch.float32,
    device=device
).view(1, 5, 1, 1)

feature_std = torch.tensor(
    [12.32, 11.47, 6.91, 0.86, 0.16],
    dtype=torch.float32,
    device=device
).view(1, 5, 1, 1)


# ============================================================
# Dataset
# ============================================================

dataset = SemanticKitti(
    root=data_root,
    sequences=[8],
    config_path=config_path
)

config = {
    "augmentation": {
        "p_flipx": 0, "p_flipy": 0,
        "p_transx": 0, "trans_xmin": 0, "trans_xmax": 0,
        "p_transy": 0, "trans_ymin": 0, "trans_ymax": 0,
        "p_transz": 0, "trans_zmin": 0, "trans_zmax": 0,
        "p_rot_roll": 0, "rot_rollmin": 0, "rot_rollmax": 0,
        "p_rot_pitch": 0, "rot_pitchmin": 0, "rot_pitchmax": 0,
        "p_rot_yaw": 0, "rot_yawmin": 0, "rot_yawmax": 0,
        "img_jitter": [0, 0, 0, 0]
    },
    "sensor": {
        "proj_h": 384,
        "proj_w": 1232,
        "proj_ht": 256,
        "proj_wt": 1024,
        "h_pad": 7,
        "w_pad": 3
    }
}

pv_loader = PerspectiveViewLoader(
    dataset=dataset,
    config=config,
    is_train=False,
    use_padding=True,
)

loader = DataLoader(
    pv_loader,
    batch_size=1,
    num_workers=4,
    shuffle=False,
    drop_last=False
)
visual_pv_loader = PerspectiveViewLoader(
    dataset=dataset,
    config=config,
    is_train=False,
    use_padding=True,
    return_uproj=True
)

visual_loader = DataLoader(
    visual_pv_loader,
    batch_size=1,
    num_workers=4,
    shuffle=False,
    drop_last=False
)

# ============================================================
# Models
# ============================================================

radiant3d_model = RADIANT3DNet(
    pcd_channels=5,
    img_channels=3,
    nclasses=20,
    base_channels=32,
    image_backbone="resnet34",
    imagenet_pretrained=False
).to(device)

radiant3d_state = torch.load(checkpoint, map_location=device)
load_result = radiant3d_model.load_state_dict(radiant3d_state, strict=False)

expected_missing = ["lidar_stream.reliability_gamma"]
assert sorted(load_result.missing_keys) == expected_missing, load_result.missing_keys
assert not load_result.unexpected_keys, load_result.unexpected_keys

radiant3d_model.eval()
print("RADIANT-3D checkpoint loaded; only expected gamma parameter is missing.")

baseline_model = PMFNet(
    pcd_channels=5,
    img_channels=3,
    nclasses=20,
    base_channels=32,
    image_backbone="resnet34",
    imagenet_pretrained=False
).to(device)

baseline_state = torch.load(baseline_checkpoint, map_location=device)
baseline_model.load_state_dict(baseline_state)
baseline_model.eval()

print("Both checkpoints loaded successfully.")

# ============================================================
# IoU metric
# ============================================================

metrics = IOUEval(20, ignore=[0])

class_names = [
    "unlabeled",
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign"

]


# ============================================================
# Validation: PMFNet vs RADIANT3DNet
# ============================================================

def evaluate_model(model, model_name, use_mask):
    metrics = IOUEval(20, ignore=[0])
    pred_counter = Counter()

    print(f"\nEvaluating {model_name} on SemanticKITTI sequence 08...")

    with torch.no_grad():
        for i, (input_feature, input_mask, input_label) in enumerate(loader):
            input_feature = input_feature.to(device)
            input_mask = input_mask.to(device)
            input_label = input_label.to(device).long()

            input_feature[:, 0:5] = (
                (input_feature[:, 0:5] - feature_mean) / feature_std
            ) * input_mask.unsqueeze(1).expand_as(input_feature[:, 0:5])

            pcd_feature = input_feature[:, 0:5]
            img_feature = input_feature[:, 5:8]

            if use_mask:
                lidar_pred, _ = model(pcd_feature, img_feature, input_mask)
            else:
                lidar_pred, _ = model(pcd_feature, img_feature)

            prediction = lidar_pred.argmax(dim=1)

            pred_counter.update(prediction.cpu().numpy().flatten().tolist())
            metrics.addBatch(prediction.cpu(), input_label.cpu())

            if (i + 1) % 100 == 0:
                print(f"{model_name}: processed {i + 1}/{len(loader)} scans")

    # Confusion matrix: rows=predictions, columns=ground truth.
    # Ignore only ground-truth class 0. Predictions of class 0
    # on valid ground-truth points must remain false negatives.
    confusion = metrics.conf_matrix.clone().double()
    confusion[:, 0] = 0

    tp = confusion.diag()
    fp = confusion.sum(dim=1) - tp
    fn = confusion.sum(dim=0) - tp

    class_iou = tp / (tp + fp + fn + 1e-15)
    class_precision = tp / (tp + fp + 1e-15)
    class_recall = tp / (tp + fn + 1e-15)

    valid_classes = slice(1, 20)
    mean_iou = class_iou[valid_classes].mean()
    mean_precision = class_precision[valid_classes].mean()
    mean_recall = class_recall[valid_classes].mean()

    overall_accuracy = (
        tp[1:].sum() / (confusion[:, 1:].sum() + 1e-15)
    )

    print(f"\n========== {model_name} RESULTS ==========")
    print(f"{'Class':<18} {'IoU':>9} {'Precision':>11} {'Recall':>10}")

    for c in range(1, 20):
        print(
            f"{class_names[c]:<18} "
            f"{class_iou[c].item() * 100:>8.2f}% "
            f"{class_precision[c].item() * 100:>10.2f}% "
            f"{class_recall[c].item() * 100:>9.2f}%"
        )

    print("-----------------------------------------------")
    print(f"mIoU:                  {mean_iou.item() * 100:.2f}%")
    print(f"Macro precision:       {mean_precision.item() * 100:.2f}%")
    print(f"Macro recall:          {mean_recall.item() * 100:.2f}%")
    print(f"Overall point accuracy:{overall_accuracy.item() * 100:.2f}%")
    print("===============================================\n")

    return {
        "mIoU": mean_iou.item() * 100,
        "mean_precision": mean_precision.item() * 100,
        "mean_recall": mean_recall.item() * 100,
        "overall_accuracy": overall_accuracy.item() * 100,
    }


baseline_results = evaluate_model(baseline_model, "Original PMFNet", False)
radiant3d_results = evaluate_model(radiant3d_model, "RADIANT3D", True)

print("\\n========== FINAL COMPARISON: PMF vs RADIANT3D ==========")
print(f"{'Metric':<22} {'PMFNet':>10} {'RADIANT3D':>12} {'Change':>10}")
for key, label in [
    ("mIoU", "mIoU"),
    ("overall_accuracy", "Overall accuracy"),
    ("mean_precision", "Macro precision"),
    ("mean_recall", "Macro recall"),
]:
    a = baseline_results[key]
    b = radiant3d_results[key]
    print(f"{label:<22} {a:>9.2f}% {b:>11.2f}% {b-a:>+9.2f}")
print("========================================================")

raise SystemExit(0)
# ================= VISUAL COMPARISON =================

class_colors = np.array([
    [255, 255, 255],  # 0: unlabeled
    [100, 150, 245],  # 1: car
    [100, 230, 245],  # 2: bicycle
    [30, 60, 150],    # 3: motorcycle
    [80, 30, 180],    # 4: truck
    [0, 0, 255],      # 5: other-vehicle
    [255, 30, 30],    # 6: person
    [255, 40, 200],   # 7: bicyclist
    [150, 30, 90],    # 8: motorcyclist
    [255, 0, 255],    # 9: road
    [255, 150, 255],  # 10: parking
    [75, 0, 75],      # 11: sidewalk
    [175, 0, 75],     # 12: other-ground
    [255, 200, 0],    # 13: building
    [255, 120, 50],   # 14: fence
    [0, 175, 0],      # 15: vegetation
    [135, 60, 0],     # 16: trunk
    [150, 240, 80],   # 17: terrain
    [255, 240, 150],  # 18: pole
    [255, 0, 0]       # 19: traffic-sign
], dtype=np.uint8)
output_dir = "RADIANT3D_comparison_visualizations"
os.makedirs(output_dir, exist_ok=True)

radiant3d_model.eval()
baseline_model.eval()

with torch.no_grad():
        target_frames = {3014, 3015}
        for i, (features, mask, labels, x_data, y_data, depth, xyz) in enumerate(visual_loader):
            if i not in target_frames:
                continue

            target_frame = i

            features = features.to(device)
            mask = mask.to(device)
            labels = labels.to(device).long()
    
            h, w = features.shape[-2:]
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16
    
            features[:, 0:5] = (
                (features[:, 0:5] - feature_mean) / feature_std
            ) * mask.unsqueeze(1)
    
            pcd = F.pad(features[:, 0:5], (0, pad_w, 0, pad_h))
            rgb = F.pad(features[:, 5:8], (0, pad_w, 0, pad_h))
            padded_mask = F.pad(mask, (0, pad_w, 0, pad_h))
    
            pmf_pred, _ = baseline_model(pcd, rgb)
            radiant3d_pred, _ = radiant3d_model(pcd, rgb, padded_mask)
    
            pmf_pred = pmf_pred.argmax(dim=1)[0, :h, :w].cpu().numpy()
            radiant3d_pred = radiant3d_pred.argmax(dim=1)[0, :h, :w].cpu().numpy()
    
            gt = labels[0].cpu().numpy()
            rgb_image = features[0, 5:8].permute(1, 2, 0).cpu().numpy()
            rgb_image = np.clip(rgb_image, 0, 1)
    
            x = x_data[0].long()
            y = y_data[0].long()
    
            xyz_points = xyz[0].cpu().numpy()

            # Retrieve camera-aligned SemanticKITTI labels
            seq_id, frame_id = dataset.parsePathInfoByIndex(i)
            _, _, instance_labels = dataset.loadDataByIndex(i)

            pointcloud, _, _ = dataset.loadDataByIndex(i)
            _, instance_keep_mask = dataset.mapLidar2Camera(
                seq_id, pointcloud[:, :3], w, h
            )
            instance_points = instance_labels[instance_keep_mask]

            assert len(xyz_points) == len(instance_points), (
                f"Point alignment failed: {len(xyz_points)} vs "
                f"{len(instance_points)}"
            )

            print(f"Frame: {seq_id}/{frame_id}")
            print(f"LiDAR points: {len(xyz_points)}")
            print(f"Aligned instance IDs: {len(instance_points)}")
            print(f"Unique instance IDs: {len(np.unique(instance_points))}")
    
            gt_points = gt[x, y]
            pmf_points = pmf_pred[x, y]
            radiant3d_points = radiant3d_pred[x, y]

            # Original RGB vs actual RADIANT-3D camera-projected predictions
            original_camera = np.asarray(dataset.loadImage(i)).copy()
            if original_camera.shape[:2] != (h, w):
                raise ValueError(
                    f"Camera image size {original_camera.shape[:2]} "
                    f"does not match prediction size {(h, w)}"
                )

            rows = x.cpu().numpy().astype(np.int32)
            cols = y.cpu().numpy().astype(np.int32)

            valid = (
                (rows >= 0) & (rows < h) &
                (cols >= 0) & (cols < w)
            )

            predicted_camera = np.zeros((h, w, 3), dtype=np.uint8)
            predicted_camera[rows[valid], cols[valid]] = (
                class_colors[radiant3d_points[valid]].astype(np.uint8)
            )

            fig_cam, axes_cam = plt.subplots(1, 2, figsize=(18, 7))
            axes_cam[0].imshow(original_camera)
            axes_cam[0].set_title("Original SemanticKITTI RGB", fontsize=16)
            axes_cam[0].axis("off")

            axes_cam[1].imshow(predicted_camera)
            axes_cam[1].set_title("RADIANT-3D Predicted Semantic Image", fontsize=16)
            axes_cam[1].axis("off")

            fig_cam.suptitle(
                f"Original vs RADIANT-3D Prediction - {seq_id}/{frame_id}",
                fontsize=18, fontweight="bold"
            )
            fig_cam.tight_layout()
            fig_cam.savefig(
                os.path.join(output_dir, f"original_vs_radiant3d_{target_frame}.png"),
                dpi=300, bbox_inches="tight"
            )
            plt.close(fig_cam)
            print("Saved original vs RADIANT-3D camera comparison.")
            different = pmf_points != radiant3d_points
            print(f"Different predictions: {different.sum()}/{len(different)}")
            print(f"Prediction difference: {different.mean() * 100:.2f}%")
            fig = plt.figure(figsize=(24, 8))
    
            def draw_3d(ax, point_labels, title):
                ax.scatter(
                    xyz_points[:, 0],
                    xyz_points[:, 1],
                    xyz_points[:, 2],
                    c=class_colors[point_labels] / 255.0,
                    s=0.5,
                    depthshade=False,
                    rasterized=True
                )
    
                ax.set_title(title, fontsize=19, fontweight="bold", pad=12)
                ax.view_init(elev=20, azim=-90)
                ax.set_box_aspect(
                    np.maximum(np.ptp(xyz_points, axis=0), 1e-3)
                )
                ax.set_axis_off()
    
            from matplotlib.patches import Circle

            # Recover the LiDAR intensity channel before normalization
            intensity = (
                features[0, 4, x, y].cpu().numpy()
                * feature_std[0, 4].item()
                + feature_mean[0, 4].item()
            )

            lo, hi = np.percentile(intensity, [2, 98])
            intensity_display = np.clip(
                (intensity - lo) / max(hi - lo, 1e-6), 0, 1
            )

            fig = plt.figure(figsize=(32, 8))

            def setup_bev(ax, title):
                ax.set_title(title, fontsize=19, fontweight="bold", pad=12)
                ax.set_xlabel("X (metres)", fontsize=11)
                ax.set_ylabel("Y (metres)", fontsize=11)
                ax.set_aspect("equal", adjustable="box")
                ax.grid(False)

            # 1. Original RGB camera image
            ax0 = fig.add_subplot(151)
            ax0.imshow(rgb_image)
            ax0.set_title("Original RGB Camera Image", fontsize=19, fontweight="bold")
            ax0.axis("off")

            # 2. Original LiDAR intensity
            ax1 = fig.add_subplot(152)
            ax1.scatter(
                xyz_points[:, 0], xyz_points[:, 1],
                c=intensity_display, cmap="gray",
                s=0.35, rasterized=True
            )
            setup_bev(ax1, "Original LiDAR Scene")

            # 2. Original PMF prediction
            ax2 = fig.add_subplot(153)
            ax2.scatter(
                xyz_points[:, 0], xyz_points[:, 1],
                c=class_colors[pmf_points] / 255.0,
                s=0.35, rasterized=True
            )
            setup_bev(ax2, "Original PMF Prediction")

            # 3. RADIANT-3D prediction
            ax3 = fig.add_subplot(154)
            ax3.scatter(
                xyz_points[:, 0], xyz_points[:, 1],
                c=class_colors[radiant3d_points] / 255.0,
                s=0.35, rasterized=True
            )
            setup_bev(ax3, "RADIANT-3D Prediction")

            # 4. Model disagreement
            ax4 = fig.add_subplot(155)
            diff_colors = np.where(
                different[:, None],
                np.array([[1.0, 0.0, 0.0]]),
                np.array([[0.55, 0.55, 0.55]])
            )

            ax4.scatter(
                xyz_points[:, 0], xyz_points[:, 1],
                c=diff_colors, s=0.4, rasterized=True
            )
            setup_bev(ax4, "Prediction Differences")

            # Find the three spatial grid cells with the most disagreements
            cell_size = 5.0
            cells = np.floor(
                xyz_points[different, :2] / cell_size
            ).astype(int)

            if len(cells):
                unique_cells, counts = np.unique(
                    cells, axis=0, return_counts=True
                )
                ranked = np.argsort(counts)[::-1]
                selected = []

                for idx in ranked:
                    cell = unique_cells[idx]
                    center_x = (cell[0] + 0.5) * cell_size
                    center_y = (cell[1] + 0.5) * cell_size

                    if all(
                        np.hypot(center_x - sx, center_y - sy) > 5.0
                        for sx, sy in selected
                    ):
                        selected.append((center_x, center_y))

                    if len(selected) == 3:
                        break

                for number, (cx, cy) in enumerate(selected, start=1):
                    ax4.add_patch(Circle(
                        (cx, cy), radius=3.0,
                        fill=False, edgecolor="red", linewidth=2.5
                    ))
                    ax4.annotate(
                        str(number), (cx, cy),
                        color="red", fontsize=12, fontweight="bold"
                    )

            print("Highlighted disagreement regions:", len(selected) if len(cells) else 0)

            from matplotlib.lines import Line2D

            class_names = [
                "Unlabeled", "Car", "Bicycle", "Motorcycle", "Truck",
                "Other vehicle", "Person", "Bicyclist", "Motorcyclist",
                "Road", "Parking", "Sidewalk", "Other ground",
                "Building", "Fence", "Vegetation", "Trunk",
                "Terrain", "Pole", "Traffic sign"
            ]

            legend_handles = [
                Line2D(
                    [0], [0], marker="o", color="w",
                    label=name,
                    markerfacecolor=class_colors[idx] / 255.0,
                    markersize=7
                )
                for idx, name in enumerate(class_names)
            ]

            fig.legend(
                handles=legend_handles,
                loc="lower center",
                ncol=10,
                fontsize=9,
                frameon=True,
                bbox_to_anchor=(0.5, -0.10)
            )

            fig.subplots_adjust(
                left=0.01, right=0.99,
                bottom=0.20, top=0.92, wspace=0.02
            )
            plt.savefig(
                os.path.join(output_dir, f"comparison_{i:03d}.png"),
                dpi=300,
                bbox_inches="tight"
            )
            plt.close(fig)

            # Fresh wide 3D semantic scene, built from actual model predictions.
            from sklearn.cluster import DBSCAN
            from matplotlib.lines import Line2D

            fig3d = plt.figure(figsize=(22, 12), facecolor="white")

            # Top row: actual camera image and actual RADIANT-3D prediction.
            ax_original = fig3d.add_axes([0.035, 0.68, 0.45, 0.25])
            ax_original.imshow(original_camera)
            ax_original.set_title("(a) Original RGB Image", fontsize=15, fontweight="bold")
            ax_original.axis("off")

            ax_prediction = fig3d.add_axes([0.515, 0.68, 0.45, 0.25])
            ax_prediction.imshow(predicted_camera)
            ax_prediction.set_title(
                "(b) RADIANT-3D Semantic Prediction",
                fontsize=15, fontweight="bold"
            )
            ax_prediction.axis("off")

            # Bottom row: 3D LiDAR scene and estimated object boxes.
            ax_bev = fig3d.add_axes([0.055, 0.055, 0.88, 0.57])

            valid = (
                np.isfinite(xyz_points).all(axis=1)
                & (radiant3d_points >= 0)
                & (radiant3d_points < len(class_colors))
            )
            scene_xyz = xyz_points[valid]
            scene_labels = radiant3d_points[valid]

            # Forward X vertical; lateral Y horizontal.
            ax_bev.scatter(
                -scene_xyz[:, 1], scene_xyz[:, 0],
                c=class_colors[scene_labels] / 255.0,
                s=2.0, alpha=0.95, linewidths=0
            )

            # Only object-like classes receive individual boxes.
            object_classes = {
                1: ("CAR", 0.9, 6, 8.0),
                2: ("BICYCLE", 0.45, 4, 3.5),
                3: ("MOTORCYCLE", 0.5, 4, 4.0),
                4: ("TRUCK", 1.3, 6, 15.0),
                5: ("OTHER VEHICLE", 0.9, 6, 10.0),
                6: ("PERSON", 0.35, 4, 1.8),
                7: ("BICYCLIST", 0.5, 4, 3.5),
                8: ("MOTORCYCLIST", 0.5, 4, 4.0)
            }

            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]

            boxes_drawn = 0

            for class_id, (name, eps, min_samples, max_length) in object_classes.items():
                points = scene_xyz[scene_labels == class_id]
                if len(points) < min_samples:
                    continue

                clusters = DBSCAN(
                    eps=eps, min_samples=min_samples
                ).fit_predict(points[:, :2])

                for cluster_id in np.unique(clusters):
                    if cluster_id == -1:
                        continue

                    cluster = points[clusters == cluster_id]
                    if len(cluster) < min_samples:
                        continue

                    # PCA gives the estimated horizontal object orientation.
                    center = cluster[:, :2].mean(axis=0)
                    centered = cluster[:, :2] - center
                    covariance = np.cov(centered, rowvar=False)
                    values, vectors = np.linalg.eigh(covariance)
                    rotation = vectors[:, np.argsort(values)[::-1]]

                    if rotation[0, 0] < 0:
                        rotation[:, 0] *= -1
                    if np.linalg.det(rotation) < 0:
                        rotation[:, 1] *= -1

                    local = centered @ rotation
                    low = np.percentile(local, 2, axis=0)
                    high = np.percentile(local, 98, axis=0)

                    z_low, z_high = np.percentile(cluster[:, 2], [2, 98])
                    length, width = high - low
                    height = z_high - z_low

                    if min(length, width, height) < 0.12:
                        continue
                    if max(length, width) > max_length:
                        continue
                    if height > 4.5:
                        continue

                    corners_local = np.array([
                        [low[0], low[1]], [high[0], low[1]],
                        [high[0], high[1]], [low[0], high[1]]
                    ])
                    corners_xy = corners_local @ rotation.T + center

                    corners = np.vstack([
                        np.column_stack([corners_xy, np.full(4, z_low)]),
                        np.column_stack([corners_xy, np.full(4, z_high)])
                    ])

                    color = class_colors[class_id] / 255.0
                    for a, b in edges:
                        ax_bev.plot(
                            -corners[[a, b], 1],
                            corners[[a, b], 0],
                            color=color, linewidth=2.0
                        )

                    boxes_drawn += 1

            # Clean 2D bird's-eye view: X-Y only, no 3D perspective.
            ax_bev.set_title(
                f"(c) RADIANT-3D 2D Bird's-Eye Semantic Scene — {boxes_drawn} estimated boxes",
                fontsize=15, fontweight="bold", pad=12
            )
            ax_bev.set_xlabel("Y (m) — lateral", fontsize=13)
            ax_bev.set_ylabel("X (m) — forward", fontsize=13)
            ax_bev.set_aspect("equal", adjustable="box")
            ax_bev.grid(True, alpha=0.22)
            ax_bev.set_facecolor("#f7f8fa")

            y_min, y_max = np.percentile(scene_xyz[:, 1], [1, 99])
            x_min, x_max = np.percentile(scene_xyz[:, 0], [1, 99])
            ax_bev.set_xlim(-y_max - 0.5, -y_min + 0.5)
            ax_bev.set_ylim(x_min - 0.5, x_max + 0.5)

            legend_items = []
            for class_id in np.unique(scene_labels):
                if class_id == 0:
                    continue
                color = class_colors[class_id] / 255.0
                class_names = {
                    1: "Car",
                    2: "Bicycle",
                    3: "Motorcycle",
                    4: "Truck",
                    5: "Other Vehicle",
                    6: "Person",
                    7: "Bicyclist",
                    8: "Motorcyclist",
                    9: "Road",
                    10: "Parking",
                    11: "Sidewalk",
                    12: "Other Ground",
                    13: "Building",
                    14: "Fence",
                    15: "Vegetation",
                    16: "Trunk",
                    17: "Terrain",
                    18: "Pole",
                    19: "Traffic Sign"
                }
                class_name = class_names.get(int(class_id), "Unknown")
                legend_items.append(
                    Line2D([0], [0], color=color, lw=2.5, label=class_name)
                )

            ax_bev.legend(
                handles=legend_items,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                title="SEMANTIC CLASSES",
                fontsize=9,
                title_fontsize=10,
                frameon=True
            )

            fig3d.suptitle(
                f"RADIANT-3D Result Visualization — {seq_id}/{frame_id}",
                fontsize=19, fontweight="bold", y=0.98
            )

            fig3d.savefig(
                os.path.join(
                    output_dir,
                    f"radiant3d_full_visualization_{target_frame}.png"
                ),
                dpi=300, bbox_inches="tight"
            )
            plt.close(fig3d)


            # ============================================================
            # NEW: Genuine PMF vs RADIANT-3D six-panel comparison
            # ============================================================
            from matplotlib.patches import Circle

            comparison_fig, comparison_axes = plt.subplots(
                2, 3, figsize=(24, 14), facecolor="white"
            )
            a_rgb, a_pmf, a_adaptive = comparison_axes[0]
            a_pmf_bev, a_adaptive_bev, a_diff_bev = comparison_axes[1]

            # Top row: same original camera frame and both model predictions.
            pmf_camera = np.zeros((h, w, 3), dtype=np.uint8)
            adaptive_camera = np.zeros((h, w, 3), dtype=np.uint8)

            valid_pixels = (
                valid
                & (pmf_points >= 0) & (pmf_points < len(class_colors))
                & (radiant3d_points >= 0) & (radiant3d_points < len(class_colors))
            )

            pmf_camera[rows[valid_pixels], cols[valid_pixels]] = (
                class_colors[pmf_points[valid_pixels]]
            )
            adaptive_camera[rows[valid_pixels], cols[valid_pixels]] = (
                class_colors[radiant3d_points[valid_pixels]]
            )

            a_rgb.imshow(original_camera)
            a_rgb.set_title("(a) Original RGB Camera", fontsize=15, fontweight="bold")
            a_pmf.imshow(pmf_camera)
            a_pmf.set_title("(b) Original PMF Prediction", fontsize=15, fontweight="bold")
            a_adaptive.imshow(adaptive_camera)
            a_adaptive.set_title("(c) RADIANT-3D V1 Prediction", fontsize=15, fontweight="bold")

            for ax in (a_rgb, a_pmf, a_adaptive):
                ax.axis("off")

            # Bottom row: identical LiDAR coordinates and identical BEV limits.
            scene_valid = (
                np.isfinite(xyz_points).all(axis=1)
                & (pmf_points >= 0) & (pmf_points < len(class_colors))
                & (radiant3d_points >= 0) & (radiant3d_points < len(class_colors))
            )
            scene_xyz_compare = xyz_points[scene_valid]
            pmf_labels_compare = pmf_points[scene_valid]
            radiant3d_labels_compare = radiant3d_points[scene_valid]
            diff_compare = pmf_labels_compare != radiant3d_labels_compare

            if len(scene_xyz_compare) == 0:
                raise ValueError("No valid aligned LiDAR points for comparison.")

            lateral = -scene_xyz_compare[:, 1]
            forward = scene_xyz_compare[:, 0]

            def draw_comparison_bev(ax, labels, title):
                ax.scatter(
                    lateral, forward,
                    c=class_colors[labels] / 255.0,
                    s=0.45, linewidths=0, rasterized=True
                )
                ax.set_title(title, fontsize=14, fontweight="bold")
                ax.set_xlabel("Y (m) — lateral")
                ax.set_ylabel("X (m) — forward")
                ax.set_aspect("equal", adjustable="box")
                ax.grid(True, alpha=0.18)
                ax.set_facecolor("#f7f8fa")
                ax.set_xlim(
                    np.percentile(lateral, 1) - 0.5,
                    np.percentile(lateral, 99) + 0.5
                )
                ax.set_ylim(
                    np.percentile(forward, 1) - 0.5,
                    np.percentile(forward, 99) + 0.5
                )

            draw_comparison_bev(a_pmf_bev, pmf_labels_compare, "(d) PMF 2D Bird's-Eye View")
            draw_comparison_bev(
                a_adaptive_bev, radiant3d_labels_compare,
                "(e) RADIANT-3D V1 2D Bird's-Eye View"
            )

            # Red = predictions differ; grey = predictions agree.
            diff_rgb = np.where(
                diff_compare[:, None],
                np.array([[1.0, 0.0, 0.0]]),
                np.array([[0.72, 0.72, 0.72]])
            )
            a_diff_bev.scatter(
                lateral, forward, c=diff_rgb,
                s=0.55, linewidths=0, rasterized=True
            )
            a_diff_bev.set_title(
                f"(f) Prediction Disagreement — {diff_compare.sum()}/{len(diff_compare)} "
                f"points ({diff_compare.mean() * 100:.2f}%)",
                fontsize=14, fontweight="bold"
            )
            a_diff_bev.set_xlabel("Y (m) — lateral")
            a_diff_bev.set_ylabel("X (m) — forward")
            a_diff_bev.set_aspect("equal", adjustable="box")
            a_diff_bev.grid(True, alpha=0.18)
            a_diff_bev.set_facecolor("#f7f8fa")
            a_diff_bev.set_xlim(a_pmf_bev.get_xlim())
            a_diff_bev.set_ylim(a_pmf_bev.get_ylim())

            # Circle the three most spatially concentrated disagreement regions.
            cell_size = 5.0
            disagreement_xy = np.column_stack([
                lateral[diff_compare], forward[diff_compare]
            ])

            if len(disagreement_xy):
                cell_ids = np.floor(disagreement_xy / cell_size).astype(int)
                unique_cells, counts = np.unique(
                    cell_ids, axis=0, return_counts=True
                )
                selected_centres = []

                for idx in np.argsort(counts)[::-1]:
                    cx = (unique_cells[idx, 0] + 0.5) * cell_size
                    cy = (unique_cells[idx, 1] + 0.5) * cell_size

                    if all(
                        np.hypot(cx - sx, cy - sy) > 5.0
                        for sx, sy in selected_centres
                    ):
                        selected_centres.append((cx, cy))

                    if len(selected_centres) == 3:
                        break

                for number, (cx, cy) in enumerate(selected_centres, 1):
                    a_diff_bev.add_patch(Circle(
                        (cx, cy), radius=3.0,
                        fill=False, edgecolor="red", linewidth=2.5
                    ))
                    a_diff_bev.annotate(
                        str(number), (cx, cy),
                        color="red", fontsize=12, fontweight="bold"
                    )

            comparison_fig.suptitle(
                f"PMF vs RADIANT-3D V1 — SemanticKITTI {seq_id}/{frame_id}",
                fontsize=20, fontweight="bold"
            )
            comparison_fig.text(
                0.5, 0.015,
                "Same RGB frame and aligned LiDAR points | Red circles mark regions "
                "with concentrated prediction disagreements — not necessarily errors.",
                ha="center", fontsize=11
            )
            comparison_fig.tight_layout(rect=[0, 0.04, 1, 0.95])

            comparison_path = os.path.join(
                output_dir, f"pmf_vs_radiant3d_comparison_{target_frame}.png"
            )
            comparison_fig.savefig(
                comparison_path, dpi=300, bbox_inches="tight"
            )
            plt.close(comparison_fig)

            print(
                f"Saved PMF vs RADIANT-3D comparison: {comparison_path} "
                f"({diff_compare.mean() * 100:.2f}% disagreement)"
            )

            print(f"Fresh 2D BEV scene generated with {boxes_drawn} estimated object boxes.")
            print(f"Saved comparison frame {i + 1}/5")
            print(f"Saved comparison frame {i + 1}/5")

print(f"All comparison images saved in: {output_dir}")
