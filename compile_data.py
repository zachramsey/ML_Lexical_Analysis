import csv

files = ['data/bless.tsv', 'data/eval.tsv', 'data/leds.tsv', 'data/shwartz.tsv', 'data/wbless.tsv']
num_rec = 0
num_dup = 0
with open('data/compiled_data.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    for file in files:
        with open(file, 'r', encoding='utf-8') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            prev = None
            # just use this to include consecutive duplicates
            # if row[0] != 'word1':
            #     writer.writerow([row[0], row[1], 1 if row[2] == 'True' else 0])
            for row in reader:
                if row != prev and row[0] != 'word1':
                    num_rec += 1
                    writer.writerow([row[0], row[1], 1 if row[2] == 'True' else 0])
                else:
                    num_dup += 1
                prev = row

print('Number of records:', num_rec)
print('Number of duplicate records:', num_dup)
print('Number of unique records:', num_rec - num_dup)
print('Data compiled into compiled_data.csv file.')