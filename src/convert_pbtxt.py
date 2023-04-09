input_txt = "KOKUYO_data/annotations/classes_en.txt"
output_txt = "KOKUYO_data/annotations/action_list.pbtxt"

lines = open(input_txt).readlines()
lines = [x.strip().split(': ') for x in lines]
text = ""
for x in lines:
    text = text + "label {\n"
    text = text + f"  name: \"{x[1]}\"\n"
    text = text + f"  label_id: {x[0]}\n"
    text = text + f"  label_type: PERSON_MOVEMENT\n"
    text = text + "}\n"

with open(output_txt, mode='w') as f:
    f.write(text)

