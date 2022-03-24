from LV0 import wine_test

def pip_install(package):
    import sys
    import subprocess
    try:
        subprocess.check_call(["sudo", sys.executable, "-m", "pip", "install", package])
        print("install success")
    except:
        print("install fail")


#와 이건 멍꿀 함수다
def custom_split(sepr_list, str_to_list):
    import re
    regular_exp = "|".join(map(re.escape, sepr_list))
    return re.split(regular_exp, str_to_list)


def delete_file_list(file_path_list):
    try:
        import os
        for path in file_path_list:
            if os.path.exists(path):  # 파일이 있으면 삭제
                os.remove(path)
    except FileNotFoundError:
        print("File Not Found")
    finally:
        del os


def find_file_list(file_path_list):
    import os
    for path in file_path_list:
        if not os.path.exists(path):
            return False

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

        import zipfile
        # 해당 파일(여기는 zip 내용물 전부 읽는다)
        with zipfile.ZipFile("./" + data_set_folder_path + file_front_name + ".zip", "r") as existing_zip:
            existing_zip.extractall("./" + data_set_folder_path)  # 해당 위치에 압축 해제

        import re
        filen = ["train", "test", "submission"]
        for i in range(len(file_path_list)):
            # 이름 := 데이터종류이름_[train|test|submission].csv로 변경
            os.renames("./" + data_set_folder_path + filen[i] + file_extension,
                       "./" + data_set_folder_path + file_front_name + "_" + custom_split(["/", ".", "_"], file_path_list[i])[-2] + file_extension)
        del re

        # 뭔가 압축 풀때 아무런 의미 없는 폴더와 하위 파일들이 나온다
        # (이유는 모르겠다 MAC에서 이런 현상이 나오는거 같은데)
        os.remove("./" + data_set_folder_path + file_front_name + ".zip")
        if os.path.exists("./" + data_set_folder_path + "__MACOSX"):
            import shutil
            shutil.rmtree("./" + data_set_folder_path + "__MACOSX", ignore_errors=True)
            del shutil


if __name__ == "__main__":
    bike_url = "https://bit.ly/3gLj0Q6"
    wine_url = "https://bit.ly/3i4n1QB"
    # stub_download_csv(wine_url, "wine", False)
    wine_test.stub()
