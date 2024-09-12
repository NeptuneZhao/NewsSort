def read():
    # BUG - FIX: To upgrade the paddlenlp to the latest version
    # -! pip install --upgrade paddlenlp !- 
    from paddlenlp.data import Stack, Tuple, Pad
    a = [1, 2, 3, 4]
    b = [3, 4, 5, 6]
    c = [5, 6, 7, 8]
    result = Stack()([a, b, c])

    print('\nStacked data: \n', result)

    a = [1, 2, 3, 4]
    b = [5, 6, 7]
    c = [8, 9]
    result = Pad(pad_val = 0)([a, b, c])

    print('\nPadded data: \n', result)

    data = [
        [[1, 2, 3, 4], [1]],
        [[5, 6, 7], [0]],
        [[8, 9], [1]]
    ]

    batchify_fn = Tuple( Pad(pad_val = 0), Stack() )
    ids, labels = batchify_fn(data)

    print('\nids: \n', ids)
    print('\nlabels: \n', labels)

if __name__ == '__main__':
    read()