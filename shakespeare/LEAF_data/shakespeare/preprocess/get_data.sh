cd ../data/raw_data

# Scarica il file dal link aggiornato
curl -O https://www.gutenberg.org/files/100/100-0.txt
mv 100-0.txt raw_data.txt

cd ../../preprocess
