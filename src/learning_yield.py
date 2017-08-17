"""
created by DucVu
"""

def learning_gener_yield():
    mylist = range(3)
    print mylist
    for i in mylist:
        yield i*i
        print ("o")
    print "finished"
if __name__ == '__main__':
    gen = learning_gener_yield()
    for i in gen:
        print i
    a = learning_gener_yield()
    for i in a:
        print i
        print ("---------")
