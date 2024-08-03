# This sis what a comment looks like
fruits = ['apple', 'orange', 'pears', 'bananas']
for fruit in fruits:
    print(fruit + ' for sale')

fruitPrices = {'apples': 2.00, 'oranges': 1.50, 'pears': 1.75} # Dictionary
for fruit, price in fruitPrices.items(): # if you do not use '.itens()' function, it may occur a value error "too many values to unpack"
    # iterate through key and value in the dictionary
    if price < 2.00:
        print('%s cost %f a pound' % (fruit, price))
    else:
        print(fruit + ' are too expensive!')