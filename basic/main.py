def pip_install(package):
    import sys
    import subprocess
    try:
        subprocess.check_call(["sudo", sys.executable, "-m", "pip", "install", package])
        print("install success")
    except:
        print("install fail")


# 와 이건 꿀함수다
def custom_split(sepr_list, str_to_list):
    import re
    regular_exp = "|".join(map(re.escape, sepr_list))
    return re.split(regular_exp, str_to_list)



def download_csv(url, type_name, force=False):
    import os
    nowPath = os.getcwd().replace("\\", "/")
    temp_path = "/temp"

    try:
        os.mkdir(nowPath + temp_path)
        import wget
        wget.download(url, out=nowPath + temp_path + "/download.zip")
        del wget

        import zipfile
        # 다운로드
        with zipfile.ZipFile(nowPath + temp_path + "/download.zip", "r") as existing_zip:
            existing_zip.extractall(nowPath + temp_path)

        os.remove(nowPath + temp_path + "/download.zip")  # 다운로드한 zip파일 삭제
        # OS별로 포멧이나 호환성이 다르기 때문에 아이콘을 생성하기 위해
        # 파일을 가져오는 과정에서 ._으로 시작되는 파일들이 생성되는 경우가 존재
        # 아래의 조건문은 그것을 삭제하는 과정
        if os.path.exists(nowPath + temp_path + "/__MACOSX"):
            import shutil
            shutil.rmtree(nowPath + temp_path + "/__MACOSX", ignore_errors=True)
            del shutil

        file_list = os.listdir(nowPath + temp_path)
        for file in file_list:
            parse = custom_split([".", "_"], file)
            old = nowPath + temp_path + "/" + file
            new = nowPath + "/data_set" + "/" + type_name + "_" + parse[-2] + "." + parse[-1]
            if os.path.exists(new):
                os.remove(new)
            os.renames(old, new)  # 이름을 바꾸면서 잘라내서 붙여넣기 작업

        del os, zipfile
        print("download successfully:" + nowPath + "/data_set")

    except:
        print("download failed")


if __name__ == "__main__":
    url_list = {"따릉이": "https://bit.ly/3gLj0Q6",
                "와인": "https://bit.ly/3i4n1QB",
                "청와대": "https://bit.ly/3l6g8j3"}
    import test as ts
    want_data_name = "청와대"
    ts.download_csv(url_list[want_data_name], want_data_name)

    # stub()
