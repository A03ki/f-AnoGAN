from ftplib import FTP
import tarfile


MVTECAD_DATASET_NAMES = ["bottle", "cable", "capsule", "carpet", "grid",
                         "hazelnut", "leather", "metal_nut",
                         "mvtec_anomaly_detection", "pill", "screw",
                         "tile", "toothbrush", "transistor", "wood", "zipper"]


class MVTecAD:
    """Download MVTec Anomaly Detection Dataset by FTP.

    Notes
    -----
    `mvtec_anomaly_detection` is the whole dataset.

    See Also
    --------
    https://www.mvtec.com/company/research/datasets/mvtec-ad/
    """
    def __init__(self):
        self.datasets = MVTECAD_DATASET_NAMES

    def download(self, dataset_name):
        dataset_name = dataset_name.lower()
        if dataset_name not in self.datasets:
            raise ValueError(f"The dataset called `{dataset_name}` "
                             "is not exist")

        with FTP("ftp.softronics.ch") as ftp:
            ftp.login(user="guest", passwd="GU.205dldo")
            ftp.cwd("mvtec_anomaly_detection")

            with open(f"{dataset_name}.tar.xz", "wb") as f:
                ftp.retrbinary(f"RETR {dataset_name}.tar.xz", f.write)

    def extract(self, dataset_name, save_path="."):
        dataset_name = dataset_name.lower()
        with tarfile.open(f"{dataset_name}.tar.xz", "r:xz") as tf:
            tf.extractall(path=save_path)
