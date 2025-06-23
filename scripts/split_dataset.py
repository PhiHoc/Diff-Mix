import os
import pandas as pd
import shutil
import argparse
from tqdm import tqdm

def split_master_dataset(master_dir, output_root, shot_sizes):
    """
    Splits a large, balanced master dataset into smaller, balanced subsets.

    Args:
        master_dir (str): Path to the master dataset directory (containing 'data' and 'meta.csv').
        output_root (str): Path to the root directory where the new subset folders will be created.
        shot_sizes (list of int): A list of shot sizes to create subsets for (e.g., [50, 100, 200]).
    """

    # 1. Tải file metadata của bộ dữ liệu master
    master_meta_path = os.path.join(master_dir, 'meta.csv')
    if not os.path.exists(master_meta_path):
        print(f"Lỗi: Không tìm thấy file 'meta.csv' trong thư mục master: {master_dir}")
        return

    print(f"Đang tải metadata từ: {master_meta_path}")
    master_df = pd.read_csv(master_meta_path)

    # Lấy danh sách tất cả các lớp từ cột 'First Directory'
    all_classes = master_df['First Directory'].unique()
    print(f"Tìm thấy {len(all_classes)} lớp trong bộ dữ liệu master.")

    # 2. Lặp qua từng kích thước "shot" để tạo bộ dữ liệu con
    for shot in shot_sizes:
        print(f"\n{'='*20}")
        print(f"Bắt đầu xử lý cho {shot} ảnh mỗi lớp")
        print(f"{'='*20}")

        # Tạo đường dẫn cho thư mục output của bộ dữ liệu con
        split_dir_name = f"master_split_shot_{shot}"
        split_output_path = os.path.join(output_root, split_dir_name)
        split_data_path = os.path.join(split_output_path, 'data')

        os.makedirs(split_data_path, exist_ok=True)
        print(f"Đã tạo thư mục output tại: {split_output_path}")

        # Danh sách để lưu các mẫu được chọn cho bộ dữ liệu con này
        subset_dfs = []

        # 3. Lặp qua từng lớp để lấy mẫu cân bằng
        for class_name in all_classes:
            # Lọc ra tất cả các hàng thuộc về lớp hiện tại
            class_df = master_df[master_df['First Directory'] == class_name]

            # Kiểm tra xem có đủ ảnh để lấy mẫu không
            if len(class_df) < shot:
                print(f"Cảnh báo: Lớp '{class_name}' chỉ có {len(class_df)} ảnh, ít hơn {shot} yêu cầu. Sẽ lấy tất cả.")
                sampled_df = class_df
            else:
                # Lấy ngẫu nhiên 'shot' mẫu từ lớp này
                sampled_df = class_df.sample(n=shot, random_state=42) # random_state để đảm bảo kết quả lặp lại

            subset_dfs.append(sampled_df)

        # 4. Gộp tất cả các mẫu đã chọn thành một DataFrame mới
        subset_final_df = pd.concat(subset_dfs, ignore_index=True)

        # Lưu file meta.csv mới cho bộ dữ liệu con
        subset_meta_path = os.path.join(split_output_path, 'meta.csv')
        subset_final_df.to_csv(subset_meta_path, index=False)
        print(f"Đã lưu file meta.csv mới cho {shot} shots ({len(subset_final_df)} hàng).")

        # 5. Sao chép các file ảnh tương ứng
        print("Bắt đầu sao chép file ảnh...")
        for _, row in tqdm(subset_final_df.iterrows(), total=len(subset_final_df), desc=f"Copying {shot} shots"):
            # Đường dẫn file ảnh gốc
            source_image_path = os.path.join(master_dir, 'data', row['Path'])

            # Đường dẫn file ảnh đích
            dest_image_path = os.path.join(split_data_path, row['Path'])

            # Tạo thư mục con cho lớp nếu chưa tồn tại
            os.makedirs(os.path.dirname(dest_image_path), exist_ok=True)

            # Sao chép file
            if os.path.exists(source_image_path):
                shutil.copy(source_image_path, dest_image_path)
            else:
                print(f"\nCảnh báo: Không tìm thấy file ảnh gốc: {source_image_path}")

        print(f"Hoàn thành xử lý cho {shot} ảnh mỗi lớp.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split a master dataset into smaller, balanced subsets.")
    parser.add_argument('--master_dir', type=str, required=True, help="Path to the master dataset directory.")
    parser.add_argument('--output_root', type=str, required=True, help="Root directory to save the new split datasets.")
    parser.add_argument('--shot_sizes', type=int, nargs='+', required=True, help="A list of shot sizes to create (e.g., 50 100 200).")

    args = parser.parse_args()

    split_master_dataset(args.master_dir, args.output_root, args.shot_sizes)