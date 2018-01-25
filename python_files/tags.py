import lda as l
import json

# lda_tags = {"ui_tags":[], "ml_tags":[], "and_tags":[], "ai_tags":[], "prg_tags":[]}
out_file = open("output.txt", "a")
data = []

# with open("data.json") as f:
#     data = json.load(f)

count = 0

with open("sports.csv") as f:
    for line in f:
        category = line.split(",")[-1]
        if ('"' in line):
            text = line.split('"')[1]
            print("hi")
            tags = l.lda(text)
            tag_text = ' '.join(tags)
            out_file.write(tag_text + "\t" + category + "\n")
        

# for i in data:
#     print("hi")
#     text = i["body"]
#     tags = l.lda(text)
#     category = i["category"]
#     tag_text = ' '.join(tags)
#     out_file.write(tag_text + "\t" + category + "\n")
