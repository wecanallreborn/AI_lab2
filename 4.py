from nltk.corpus import brown

def chunker(input_data, N):
    words = input_data.split(' ')
    return [' '.join(words[i:i + N]) for i in range(0, len(words), N)]


N, M = 12000, 700
text = ' '.join(brown.words()[:N])
chunks = chunker(text, M)
print("Количество чанков:", len(chunks))
for i, c in enumerate(chunks[:20]):
    print(f"Chunk {i}: {c[:50]}...")