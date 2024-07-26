import csv
import json

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification

from dataset import encode_entity, ENTITY_CLASSES, RELATION_CLASSES, POSSIBLE_ENTITIES_RELATIONS
from utils import clean_text

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer._convert_token_to_id('[PAD]'),
        type_vocab_size=len(ENTITY_CLASSES) * 2,
        num_labels=len(RELATION_CLASSES)

    )
    model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased",
                                                          config=config,
                                                          ignore_mismatched_sizes=True)
    dt = torch.load('output/last_model.pt', map_location=device)
    model.load_state_dict(dt['model'])
    model.to(device)
    model.eval()

    df = pd.read_csv('data/test.csv')
    df = df.set_index("id")
    df.entities = df.entities.apply(json.loads)

    header = ['id', 'relations']
    rows = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
        relations = []

        original_text = row["text"]
        cleaned_text = clean_text(row["text"])
        tokens = tokenizer.tokenize(cleaned_text)
        ids = tokenizer.encode(cleaned_text)

        for first_entity in row['entities']:
            first_entity_encodings = encode_entity(original_text, tokens, first_entity)
            for second_entity in row['entities']:
                second_entity_encodings = encode_entity(original_text, tokens, second_entity, shift=len(ENTITY_CLASSES))
                merged_encodings = [0] + [max(x, y) for x, y in
                                          zip(first_entity_encodings, second_entity_encodings)] + [0]

                with torch.no_grad():
                    inputs = {
                        "input_ids": torch.tensor(ids).unsqueeze(0).to(device),
                        "attention_mask": (torch.tensor(ids) != config.pad_token_id).unsqueeze(0).to(device),
                        "token_type_ids": torch.tensor(merged_encodings).unsqueeze(0).to(device),
                    }
                    outputs = model(**inputs)
                    logits = outputs.logits[0].cpu().numpy()
                    predicted_relation = RELATION_CLASSES[logits.argmax()]

                    if predicted_relation != 'NO_RELATION':
                        if f"{first_entity['type']} {predicted_relation} {second_entity['type']}" in POSSIBLE_ENTITIES_RELATIONS:
                            relations.append([first_entity['id'], predicted_relation, second_entity['id']])

        rows.append([index, relations])

    with open('submission.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


