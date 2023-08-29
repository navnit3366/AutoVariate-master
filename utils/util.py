from PIL import Image
import os
from pathlib import Path
import shutil


class auto_util():

    def __init__(self):
        super().__init__()

    def __get_project_root(self) -> Path:
        return Path(__file__).parent.parent

    def __find_files(self, filename, search_path):
        result = []
        for root, dir, files in os.walk(search_path):
            if filename in files:
                result.append(os.path.join(root, filename))
        return result
    
    def erase_dir(self, dir):
        shutil.rmtree(dir, ignore_errors=True, onerror=None)
            
    def list_sub_dir(self, dir): 
        rootdir = dir
        return os.listdir(rootdir) 
 

    def show_module_details(self):
        os.system('pipdeptree -p autovariate --graph-output png > graph.png')
        img = Image.open('graph.png')
        img.show()
        graph = self.__find_files("graph.png", self.__get_project_root())[0]
        os.remove(graph)

    
    

    



            
 


