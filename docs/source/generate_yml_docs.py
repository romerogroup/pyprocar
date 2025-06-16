import os
from glob import glob
import yaml
from pathlib import Path

def generate_documentation_from_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    tmp_filename=yaml_path.split(os.sep)[-1].split('.')[0]
    # Begin the .rst document
    doc = []

    filename=tmp_filename.replace('_',' ')
    doc.append(filename + " plotting Options")
    doc.append("=====================================================")
    for key, details in config.items():
        # Key as the section title
        doc.append(key)
        doc.append('-' * len(key))  # Underline for section title
        doc.append('')

        for detail_key, detail_value in details.items():
            # Print the details
            doc.append(f":{detail_key}: {detail_value}")
            doc.append('')

        doc.append('')  # Add a space between sections

    return '\n'.join(doc)

if __name__ == '__main__':
    # Define the path to the YAML file and the output .rst file
    
    FILE = Path(__file__).resolve()
    source_dir=str(FILE.parents[1])
    project_dir=str(FILE.parents[2])

    # Write to the .rst file
    filepaths=glob(project_dir + '/pyprocar/cfg/*.yml')

    for filepath in filepaths:
        filename=os.path.basename(filepath)
        if filename == 'package.yml':
            continue
        doc_content = generate_documentation_from_yaml(filepath)
        filename=filepath.split(os.sep)[-1].split('.')[0]
        rst_path = source_dir + '/source/api/cfg/' + filename + '.rst'
        if os.path.exists(rst_path):
            os.remove(rst_path)
        with open(rst_path, 'w') as rst_file:
            rst_file.write(doc_content)

        print(f"Documentation generated and saved to {rst_path}")