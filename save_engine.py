from llama_index.core.tools import FunctionTool
import os

save_file = os.path.join('data', 'saved.txt')

def save_to_file(note):
    if not os.path.exists(save_file):
        open(save_file, 'w')
        
    with open(save_file, 'a') as file:
        file.writelines([note + '\n'])

    return f"Note saved to {save_file}"

save_engine = FunctionTool.from_defaults(
    fn=save_to_file,
    name="save first aid",
    description="this tool can save the first aid instructions to a file for future reference of the user",
)