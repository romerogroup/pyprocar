import os
import shutil
import gdown

# TODO Zip file in google drive then download
# TODO Save dir path to temp


examples_dict = {"Fe" :
                    {"qe": 
                        {
                        'non-colinear':
                            {
                            'bands':'1Eo1VUuT6rmEKMy2-NtYo2kVH5Vc4PeFJ',
                            'dos':'1y1Q0w31iN_HAUDnm4lcTdqPfWKggYDRE',
                            'fermi':'1XLiYwN_W3AFAzbGOGfGVn-ha5tXFwYLw'},
                        'non-spin-polarized':
                            {
                            'bands':'1__DI_iqS3zFVerDMh9WjqXy8oE2XaJkK?usp=share_link',
                            'dos':'1nuczS9sHhLn8UXSeoF0mS4XwxtZUEcgW',
                            'fermi':'1hjGeVXfNoqSDUMBc5TF0qzgGUfgaTg0s'},
                        'spin-polarized-colinear':
                            {
                            'bands':'1DdkLmwXQZL4S0VXiht6XQmgGPP8h_HPZ',
                            'dos':'186oZvQEh6AYre2h6FjzGQ8AWIcc50KGL',
                            'fermi':'1rF70CtYDapE1WdHATpsvUFyKAabB_Tn5'}
                        },
                    "vasp": 
                        {
                        'non-colinear':
                            {
                            'bands':'1y6Ww79kGd8hUbqcZWI0RauzfVWlhwhf7',
                            'dos':'1laWtxShW7XQUpmofOk0tX4rwAghuC4II',
                            'fermi':'1Kk56kv2pPqK4nqiTnhDm8FeTfjMoOi7P'},
                        'non-spin-polarized':
                            {
                            'bands':'1Lt-u4hFT5k97kFz9-0WY5V_XK_o2oYPl',
                            'dos':'1Ipsz1ya9Zm_k8u6hDyCA9-COtGyyEOLN',
                            'fermi':'1pEK9Q7DfzgHk1DCW3JnyHyXLcubw0sil'},
                        'spin-polarized-colinear':
                            {
                            'bands':'1Be3OSRmf3JcACA5aA8GfZGDgJdu5beRH',
                            'dos':'1EDxuW7JjIk9Q-gvOrAY9QFxPOGQt_V-W',
                            'fermi':'1BzALZKEC19mxOYgMPJ7XRjQTNQmBZKCe'}
                        },

                    "abinit": 
                        {
                        'non-colinear':
                            {
                            'bands':'',
                            'dos':'',
                            'fermi':''},
                        'non-spin-polarized':
                            {
                            'bands':'1Re6N2rj9AnzefmJOYLeUecMHbH2eMRWu',
                            'dos':'',
                            'fermi':''},
                        'spin-polarized-colinear':
                            {
                            'bands':'',
                            'dos':'',
                            'fermi':''}
                        }


                    },

                "BiSb_monolayer" :
                    {"vasp": 
                        {
                        'non-colinear':
                            {
                            'bands':'',
                            'dos':'',
                            'fermi':'192XJLLpd7knvazhJPbhcV0Jb75NZM7OF'},


                    },
                }
    }


def download_examples(save_dir=''):
    if save_dir != '':
        output = f"{save_dir}{os.sep}examples.zip"
        to = f"{save_dir}{os.sep}examples{os.sep}"
    else:
        output='examples.zip'
        to = f"examples{os.sep}"
        
    gdown.download(id="1AAcJ17ghTVcw_nRICX5IAwhqmtaRP6fd", output=output)
    gdown.extractall(output, to = to)

def download_dev_data(examples_dirname: str = 'examples'):
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_dir, 'data')

    material_name = 'Fe'

    print('Storing development data in', data_dir)
    if not os.path.exists(data_dir):
        os.mkdirs(data_dir) 

    # output = f"{data_dir}{os.sep}{examples_dirname}"
    to = f"{data_dir}{os.sep}{examples_dirname}{os.sep}{material_name}"

    print('___Starting download___')
    url  = f'https://drive.google.com/drive/folders/1FQ5suC2e-Wp9LfWQqeb_2pRJDO00QQvQ'
    gdown.download_folder(url=url, output=to,use_cookies=False,remaining_ok=True)

    print('___Download Finished___')
    return None 

def download_example(material: str,
                    code: str, 
                    spin_calc_type: str,
                    calc_type: str,
                    save_dir: str ='' ):

    materials = list(examples_dict.keys())
    codes = list(examples_dict['Fe'].keys())
    spin_calc_types = list(examples_dict['Fe']['qe'].keys())
    calc_types = list(examples_dict['Fe']['qe']['non-spin-polarized'].keys())

    if material not in materials:
        raise Exception(f"material must be in {materials}")
    if code not in codes:
        raise Exception(f"code must be in {codes}")
    if spin_calc_type not in spin_calc_types:
        raise Exception(f"spin_calc_type must be in {spin_calc_types}")
    if calc_type not in calc_types:
        raise Exception(f"calc_type must be in {calc_types}")

    url  = f'https://drive.google.com/drive/folders/{examples_dict[material][code][spin_calc_type][calc_type]}'

    dir_name = f'{material}{os.sep}{code}{os.sep}{spin_calc_type}{os.sep}'


    
    if save_dir != '':
        output = f"{save_dir}{os.sep}{dir_name}.zip"
        to = f"{save_dir}{os.sep}{dir_name}{os.sep}"
    else:
        output=f'{dir_name}.zip'
        to = f"{dir_name}{os.sep}"

    print(f"Saving to : {to} ")

    gdown.download_folder(url=url, output=to,use_cookies=False,remaining_ok=True)

    final_dir = to + calc_type
    return final_dir

