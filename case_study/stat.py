import xml.dom.minidom
import numpy as np

words = []
sens = []
texts = set()

dom = xml.dom.minidom.parse('topics2014.xml')
root = dom.documentElement
for topic in root.childNodes:
    if topic.nodeName != "topic":
        continue
    for description in topic.childNodes:
        if description.nodeName != "description":
            continue
        text = description.childNodes[0].data
        texts.add(text)
        sens.append(len(text.strip()[:-1].split('.')))
        words.append(len(text.strip().split()))

dom = xml.dom.minidom.parse('topics2015A.xml')
root = dom.documentElement
for topic in root.childNodes:
    if topic.nodeName != "topic":
        continue
    for description in topic.childNodes:
        if description.nodeName != "description":
            continue
        text = description.childNodes[0].data
        if text not in texts:
            texts.add(text)
            sens.append(len(text.strip()[:-1].split('.')))
            words.append(len(text.strip().split()))


dom = xml.dom.minidom.parse('topics2015B.xml')
root = dom.documentElement
for topic in root.childNodes:
    if topic.nodeName != "topic":
        continue
    for description in topic.childNodes:
        if description.nodeName != "description":
            continue
        text = description.childNodes[0].data
        if text not in texts:
            texts.add(text)
            sens.append(len(text.strip()[:-1].split('.')))
            words.append(len(text.strip().split()))


dom = xml.dom.minidom.parse('topics2016.xml')
root = dom.documentElement
for topic in root.childNodes:
    if topic.nodeName != "topic":
        continue
    for description in topic.childNodes:
        if description.nodeName != "description":
            continue
        text = description.childNodes[0].data
        if text not in texts:
            texts.add(text)
            sens.append(len(text.strip()[:-1].split('.')))
            words.append(len(text.strip().split()))


print(np.mean(sens))
print(np.mean(words))
print(len(sens))
import ipdb; ipdb.set_trace()