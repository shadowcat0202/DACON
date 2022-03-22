def pip_install(package):
    import sys
    import subprocess
    try:
        subprocess.check_call(["sudo", sys.executable, "-m", "pip", "install", package])
        print("install success")
    except:
        print("install fail")



def download_csv(url, force):
    import os.path

    download_file_name = url.split("/")[-1] + ".zip"

    # 데이터 파일 존재 여부
    if not os.path.exists("./train.csv") and not os.path.exists("./test.csv"):
        if force:
            print("train.csv, test.csv 원본으로 새로 다운로드")
        else:
            print("train.csv, test.csv 파일이 없어서 새로 다운로드")

        try:
            import wget
        except:
            pip_install("wget")
            import wget
        print("import wget")

        if not os.path.exists("./" + download_file_name + ".zip"):
            wget.download(url, download_file_name + ".zip")
            print("\nfile download")

        import zipfile
        with zipfile.ZipFile(download_file_name + ".zip", "r") as existing_zip:  # 해당 파일(여기는 zip 내용물 전부 읽는다)
            existing_zip.extractall("./")  # 해당 위치에 압축 해제

    if os.path.exists("./" + download_file_name + ".zip"):
        os.remove("./" + download_file_name + ".zip")

    print("다운로드 완료")
    del wget


def read_csv(file_name):
    import pandas
    data = pandas.read_csv("./" + file_name + ".csv")
    del pandas
    return data


def predict_to_csv(pred, file_name):
    import os
    import pandas as pd
    if os.path.exists("./submission.csv"):
        submission = pd.read_csv("./submission.csv")
        submission["count"] = pred
        submission.to_csv(file_name + ".csv", index=False)
    else:
        print("submission.csv file is not exists")