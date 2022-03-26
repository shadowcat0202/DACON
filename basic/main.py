from LV0 import wine_test

def pip_install(package):
    import sys
    import subprocess
    try:
        subprocess.check_call(["sudo", sys.executable, "-m", "pip", "install", package])
        print("install success")
    except:
        print("install fail")


<<<<<<< HEAD
# 와 이건 꿀함수다
=======
#와 이건 멍꿀 함수다
>>>>>>> 4db74771132a02f9efa0cbd511bac3865da6dee6
def custom_split(sepr_list, str_to_list):
    import re
    regular_exp = "|".join(map(re.escape, sepr_list))
    return re.split(regular_exp, str_to_list)



def download_csv(url, type_name, force=False):
    import os
    nowPath = os.getcwd().replace("\\", "/")
    temp_path = "/temp"

<<<<<<< HEAD
    try:
        os.mkdir(nowPath + temp_path)
        import wget
        wget.download(url, out=nowPath + temp_path + "/download.zip")
        del wget
=======
    del os
    return True


def stub_download_csv(url, file_front_name, force):
    import os  # 파일 조작
    data_set_folder_path = "data_set/"
    file_extension = ".csv"
    file_path_list = [
        "./" + data_set_folder_path + file_front_name + "_train" + file_extension,
        "./" + data_set_folder_path + file_front_name + "_test" + file_extension,
        "./" + data_set_folder_path + file_front_name + "_submission" + file_extension
    ]

    if force or not find_file_list(file_path_list):  # 하나라도 없거나 강제가 붙는다면 관련 데이터 지우고 다시 다운로드
        delete_file_list(file_path_list)

        if not os.path.exists("./" + data_set_folder_path + file_front_name + ".zip"):  # zip 파일이 없다면 다운로드
            try:
                import wget
            except:
                pip_install("wget")
                import wget
            wget.download(url=url, out="./" + data_set_folder_path + file_front_name + ".zip")  # {@parm[1] 이름으로 다운}
            del wget
>>>>>>> 4db74771132a02f9efa0cbd511bac3865da6dee6

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
<<<<<<< HEAD
    url_list = {"따릉이": "https://bit.ly/3gLj0Q6",
                "와인": "https://bit.ly/3i4n1QB",
                "청와대": "https://bit.ly/3l6g8j3"}
    import test as ts
    want_data_name = "청와대"
    ts.download_csv(url_list[want_data_name], want_data_name)

    # stub()
=======
    bike_url = "https://bit.ly/3gLj0Q6"
    wine_url = "https://bit.ly/3i4n1QB"
    # stub_download_csv(wine_url, "wine", False)
    wine_test.stub()
>>>>>>> 4db74771132a02f9efa0cbd511bac3865da6dee6
