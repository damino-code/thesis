import re

with open('/storage/home/amine/thesis/Llama-3.2-8B-Instruct-Q4_K_M/separate_hatespeech_multiattribute/src/single_attribute_analyzer.py', 'r') as f:
    content = f.read()

def_analyze = '    def analyze_attribute(self, text, attribute, comment_id=None):'
old_impl_start = content.find(def_analyze)
old_impl_end = content.find('        try:', old_impl_start)

# The part we want to extract is from old_impl_start + len(def_analyze) to old_impl_end
extracted = content[old_impl_start + len(def_analyze):old_impl_end]

new_methods = f'''    def build_prompt(self, text, attribute, comment_id=None):
{extracted}        return full_prompt

    def analyze_attribute(self, text, attribute, comment_id=None):
        """Analyze a comment for a single attribute"""
        full_prompt = self.build_prompt(text, attribute, comment_id)

'''

content = content[:old_impl_start] + new_methods + content[old_impl_end:]

with open('/storage/home/amine/thesis/Llama-3.2-8B-Instruct-Q4_K_M/separate_hatespeech_multiattribute/src/single_attribute_analyzer.py', 'w') as f:
    f.write(content)
print("Patched successfully")
