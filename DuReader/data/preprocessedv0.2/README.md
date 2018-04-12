# MRC2018 Dataset (DuReader v2.0)
##Directory
>**trainset:** preprocessed data of training set.

>**devset:** preprocessed data of development set.

>**test1set:** preprocessed data of test set 1 (will be released in other pack).

>**test1set:** preprocessed data of test set 1 (will be released in other pack).

>**evaluation_metric:** introduction of new evaluation metric and evaluation scripts.

##Raw Data Format
Here is an example of the raw data:
```
{
    "question_id": 186358,
    "question_type": "YES_NO",
    "question": "上海迪士尼可以带吃的进去吗",
    "documents": [
        {
            "paragraphs": ["text paragraph 1", "text paragraph 2"],
            "title": "上海迪士尼可以带吃的进去吗",
            "is_selected": True
        },
        # ...
    ],
    "answers": [
        "完全密封的可以，其它不可以。",                                        # answer1
        "可以的，不限制的。只要不是易燃易爆的危险物品，一般都可以带进去的。",  # answer2
        "罐装婴儿食品、包装完好的果汁、水等饮料及包装完好的食物都可以带进乐园，但游客自己在家制作的食品是不能入园，因为自制食品有一定的安全隐患。"        # answer3
    ]
    "yesno_answers": [
        "Depends",                      # corresponding to answer 1
        "Yes",                          # corresponding to answer 2
        "Depends"                       # corresponding to asnwer 3
    ]
}
```
**"question_id"** is the uniq id for each data example.
**"question_type"** provides 3 question types: **"DESCRIPTION", "YES_NO", "ENTITY"**.
For each 'YES_NO' question,  there is a **"yesno_answers"** field which contains opinion types ("YES", "NO", "DEPENDS") to corresponding answers. 
For each 'ENTITY' question, there is an **'entity_answers'** field containing a list of entity list, and each of the entity list contains named entities extracted from corresponding 'answer' sentences.
For all question types, **"documents"** field contains at most 5 documents related to the question, and we segment each document into a list of paragraphs and store them in "paragraphs" field. And each web page's title is stored in **"title"** field.
**"is_selected"** indicates whether the annotator referred to this document when summarizing the answers. If it is set to False, the annotator didn't choose this document as a reference.

##Data Preprocessing Version
For the convenience of the participants, we provide official preprocessed data.
**But we strongly encourage participants to exploit better preprocessing strategy to make results better. **
Here is an example of preprocessed data:
```
{
    "question_id": 186358,
    "question_type": "YES_NO",
    "question": "上海迪士尼可以带吃的进去吗",
    "segmented_question": ["上海", "迪士尼", "可以", "带", "吃的", "进去", "吗"],
    "documents": [
        {
            "paragraphs": ["text paragraph 1", "text paragraph 2"],
            "segmented_paragraphs": [["tokens of paragraph1"], ["tokens of paragraph2"]],
            "title": "上海迪士尼可以带吃的进去吗",
            "segmented_title": ["上海", "迪士尼", "可以", "带", "吃的", "进去", "吗"],
            "is_selected": True, # empty for test set
            "most_related_para": 0, # empty for test set
        },
        # ...
    ],
    "answers": [
        "完全密封的可以，其它不可以。",                                        # answer1
        "可以的，不限制的。只要不是易燃易爆的危险物品，一般都可以带进去的。",  # answer2
        "罐装婴儿食品、包装完好的果汁、水等饮料及包装完好的食物都可以带进乐园，但游客自己在家制作的食品是不能入园，因为自制食品有一定的安全隐患。"        # answer3
    ]
    "answer_docs": [0],
    "answer_spans": [[0, 15]]
    "fake_answers": ["完全密封的可以，其他不可以。"],
    "match_scores": [1.00],
    "segmented_answers": [
        ["完全", "密封", "的", "可以", "，", "其它", "不可以", "。"],
        ["tokens for answer2"],
        ["tokens for answer3"],

    "yesno_answers": [
        "Depends",                      # corresponding to answer 1
        "Yes",                          # corresponding to answer 2
        "Depends"                       # corresponding to asnwer 3
    ]
}
```

To make it easier for researchers to use the Dataset, we also release the preprocessed data. The preprocessing mainly does the following things:
**1. Word segmentation.** We segment all questions, answers, document titles and paragraphs into Chinese words, and the results are stored with a new field which prefix the corresponding field name with "segmented_". For example, the segmented question is stored in "segmented_question".
**2. Answer paragraph targeting.** In DuReader dataset, each question has up to 5 related documents, and the average document length is 394, since it is too heavy to feed all 5 documents into popular RC models, so we previously find the most answer related paragraph that might contain an answer for each document. And we replace original documents with the most related paragraphs  in our baseline models. The most related paragraphs are selected according to highest recall of the answer tokens of each document, and the index of the selected paragraph of each document is stored in "most_related_para".
**3. Locating answer span.** For many popular RC models, an answer span is required in training. Since the original DuReader dataset doesn't provide the answer span, we provide a simple answer span locating strategy  for convenience in our preprocess code as an optional preprocess strategy. In the strategy, we match real answer with each documents, then search the substring with maximum F1-score of the real answers, and use the span of substring as the candidate answer span. For each question we find single span as candidate, and store it in the **"answer_spans"** field, the corresponding substring spanned by answer span is stored in **"fake_answers"**, the recall of the answer span of the real answer is stored in **"match_scores"**, and the document index of the answer span is stored in "answer_docs".

Except for word segmentation, the rest of the preprocessing strategy is implemented in https://github.com/baidu/DuReader `utils/preprocess.py`


