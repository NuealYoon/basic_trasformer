from torchtext.datasets import WikiText2

if __name__ == '__main__':

    train, val, test = WikiText2()

    print(train.__next__())

    for data in train:
        print(data)

    for data in test:
        print(data)



