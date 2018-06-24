import  sys
import  os

text_help_path = os.path.join(os.getcwd(),'text_help.py')
print(sys.path)
print(text_help_path)
sys.path.append(text_help_path)

print(sys.path)
import text_helpers

text_helpers.text()

# sys.path.append()