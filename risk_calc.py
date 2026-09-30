def count_words(filename, words):
    with open(filename, 'r') as file:
        text = file.read().lower()
    counts = {word: text.split().count(word) for word in words}
    return counts

def main():
    filename = 'safe.txt'
    words = ['safe', 'unsafe']
    counts = count_words(filename, words)

    for word, count in counts.items():
        print(f"The word '{word}' appears {count} times in the file.")

    # print percentage of safe
    total = sum(counts.values())
    safe = counts['safe']
    unsafe = counts['unsafe']
    safe_percentage = safe / total * 100
    unsafe_percentage = unsafe / total * 100
    print(f"Safe: {safe_percentage:.2f}%")
    print(f"Unsafe: {unsafe_percentage:.2f}%")

if __name__ == '__main__':
    main()
