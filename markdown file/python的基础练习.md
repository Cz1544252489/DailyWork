```python
x[1:, ::2] = -99
x
```

```python
list = [1,2,3,4,5]
print(list)
print(list[-4:-1])
print(list[-1:-4])
print(list[0:5])
print(list[0:4])
print(list[::2])
print(list[1::2])

list1 = [6,7]
print([list,list1])
print(list+list1)
list2 = list + list1
list2.append(8)
print(list2)
print(list2.pop())
list2.reverse()
print(list2)
list2.extend(list1)
print(list2)
```

```python
tuple = ( 'abcd', 786 , 2.23, 'runoob', 70.2  )
tinytuple = (123, 'runoob')

print(tuple)
print(tuple[0])
print(tuple[-1])
print(tuple[2:])
print(tuple[2::-1])
print(tinytuple * 2)
print(tuple + tinytuple)
print((tuple,tinytuple))
print([tuple,tinytuple])
print(tuple,tinytuple)
```

```python
thisset = set(("Google", "Runoob", "Taobao"))
print(thisset)
print(list(thisset))
```

```python
#!/usr/bin/python3
 
tinydict = {'Name': 'Runoob', 'Age': 7, 'Class': 'First'}
 
print("tinydict['Name']: ", tinydict['Name'])
print("tinydict['Age']: ", tinydict['Age'])
```

```python
dict = {'12':'23','aa':'cc'}
print(dict.items())
print(dict.keys())
print(dict.values())
```

```python
def sma(a,*b,**c):
    print(c)

sma("a")
sma("a",("b"))
sma("a",c={"a":"b"})

def saa(**b):
    print(b)

saa(c="d",f="z")

z = lambda x,y: print(x+y)
z(1,2)
```

```python
dict={"a":1,"b":2,"c":3}
for i in dict.keys():
    print(dict[i])

for i,v in dict.items():
    print(i,v)
```

```python
print('{}和你'.format(12))
```

```python
#!/usr/bin/python3
 
class MyClass:
    """一个简单的类实例"""
    i = 12345
    def f(self):
        return 'hello world'
 
# 实例化类
x = MyClass()
 
# 访问类的属性和方法
print("MyClass 类的属性 i 为：", x.i)
print("MyClass 类的方法 f 输出为：", x.f())
```

```python
#!/usr/bin/python3
 
class Complex:
    def __init__(self, realpart, imagpart):
        self.r = realpart
        self.i = imagpart
x = Complex(3.0, -4.5)
print(x.r, x.i)   # 输出结果：3.0 -4.5
```

```python
#!/usr/bin/python3
 
#类定义
class people:
    #定义基本属性
    name = ''
    age = 0
    #定义私有属性,私有属性在类外部无法直接进行访问
    __weight = 0
    #定义构造方法
    def __init__(self,n,a,w):
        self.name = n
        self.age = a
        self.__weight = w
    def speak(self):
        print("%s 说: 我 %d 岁。" %(self.name,self.age))
 
# 实例化类
p = people('runoob',10,30)
p.speak()
```

```python
#!/usr/bin/python3
 
class Site:
    def __init__(self, name, url):
        self.name = name       # public
        self.__url = url   # private
 
    def who(self):
        print('name  : ', self.name)
        print('url : ', self.__url)
 
    def __foo(self):          # 私有方法
        print('这是私有方法')
 
    def foo(self):            # 公共方法
        print('这是公共方法')
        self.__foo()
 
x = Site('卓', 'www.cz123.top')
x.who()        # 正常输出
x.foo()        # 正常输出
x.name
# x.__foo()      # 报错
```





