import os
import pandas as pd
import random

# --- CẤU HÌNH ---
# Đường dẫn đến thư mục output của bạn
DATASET_PATH = "/content/drive/MyDrive/RareAnimal/Turtle/diffmix_sample500/turtle/shot100_diff-mix_fixed_0.8_500samples"
TARGET_COUNT = 500
# ----------------

def balance_dataset():
    meta_csv_path = os.path.join(DATASET_PATH, 'meta.csv')
    data_folder_path = os.path.join(DATASET_PATH, 'data')

    if not os.path.exists(meta_csv_path):
        print(f"Lỗi: Không tìm thấy file meta.csv tại {meta_csv_path}")
        return

    print(f"Đang đọc file metadata từ: {meta_csv_path}")
    df = pd.read_csv(meta_csv_path)

    # Cột 'First Directory' chứa tên lớp nguồn, cũng là tên thư mục
    class_counts = df['First Directory'].value_counts().to_dict()
    all_classes = df['First Directory'].unique().tolist()

    print("\n--- Phân tích số lượng ảnh hiện tại ---")
    for cls_name, count in class_counts.items():
        print(f"Lớp: {cls_name}, Số lượng: {count}")

    print("\n--- Bắt đầu quá trình cân bằng ---")

    rows_to_keep = []
    tasks_to_generate = {}

    for class_name in all_classes:
        current_count = class_counts.get(class_name, 0)
        delta = TARGET_COUNT - current_count

        class_df = df[df['First Directory'] == class_name]

        if delta == 0:
            # Đã đủ 500 ảnh
            print(f"Lớp '{class_name}' đã đủ {TARGET_COUNT} ảnh. Bỏ qua.")
            rows_to_keep.append(class_df)
        elif delta < 0:
            # Thừa ảnh, cần xóa bớt
            num_to_delete = abs(delta)
            print(f"Lớp '{class_name}' đang thừa {num_to_delete} ảnh. Bắt đầu xóa ngẫu nhiên...")

            rows_to_delete = class_df.sample(n=num_to_delete)

            for _, row in rows_to_delete.iterrows():
                file_path_to_delete = os.path.join(data_folder_path, row['Path'])
                if os.path.exists(file_path_to_delete):
                    os.remove(file_path_to_delete)

            # Giữ lại các hàng không bị xóa
            df_kept = class_df.drop(rows_to_delete.index)
            rows_to_keep.append(df_kept)
            print(f"Đã xóa {num_to_delete} ảnh. Còn lại: {len(df_kept)} ảnh.")
        else: # delta > 0
            # Thiếu ảnh, cần sinh thêm
            print(f"Lớp '{class_name}' đang thiếu {delta} ảnh.")
            tasks_to_generate[class_name] = delta
            rows_to_keep.append(class_df)

    # Tạo DataFrame mới từ các hàng đã được giữ lại
    final_df = pd.concat(rows_to_keep, ignore_index=True)

    # Lưu vào một file meta mới để an toàn
    new_meta_path = os.path.join(DATASET_PATH, 'meta_balanced.csv')
    final_df.to_csv(new_meta_path, index=False)
    print(f"\nĐã xử lý xong! File metadata mới đã được lưu tại: {new_meta_path}")

    if tasks_to_generate:
        print("\n--- CÁC TÁC VỤ CẦN THỰC HIỆN TIẾP THEO ---")
        print("Bạn cần chạy lại script sample_mp.py để sinh bổ sung các ảnh sau:")
        for class_name, num_missing in tasks_to_generate.items():
            print(f"  - Sinh thêm {num_missing} ảnh cho lớp nguồn (source class): '{class_name}'")

if __name__ == "__main__":
    balance_dataset()