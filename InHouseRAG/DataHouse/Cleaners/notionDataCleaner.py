import json
import os
from pathlib import Path

# Should enable multiprocessing, so that different files can be cleaned parallely
class NotionDataCleaner:
    def __init__(self,output_dir:str,items_to_remove:list):
        self.output_dir = output_dir
        self.filepaths = [str(file) for file in Path(output_dir).rglob('*') if file.is_file()]

        self.items_to_remove = items_to_remove

    def remove_element_from_json(self):
     # Read the file and load its contents as JSON
     for filepath in self.filepaths:
         with open(filepath, 'r') as file:
            data = json.load(file)
            for item_to_remove in self.items_to_remove:
                updated_data = [item for item in data if
                            item['type'] != item_to_remove]
                if (len(data)-len(updated_data))!=0:
                    print(str(len(data)-len(updated_data))+ ' items with type '+str(item_to_remove)+
                          'are removed from '+filepath)
                # Write the updated data back to the file
            with open(filepath, 'w') as file:
                json.dump(updated_data, file, indent=4)