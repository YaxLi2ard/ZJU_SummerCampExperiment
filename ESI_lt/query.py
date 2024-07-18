import os


def count_files_in_subfolders(root_folder):
    """
    统计指定文件夹中的每个子文件夹的文件数量。

    参数:
    root_folder (str): 根文件夹的路径。
    """
    subfolders = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]

    for subfolder in subfolders:
        subfolder_path = os.path.join(root_folder, subfolder)
        num_files = len([f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))])
        print(f"子文件夹 '{subfolder}' 中的文件总数为: {num_files}")


if __name__ == "__main__":
    root_folder = './ES-imagenet-0.18/train/'
    count_files_in_subfolders(root_folder)
