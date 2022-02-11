# another nbt.py but with pickle problem fixed (very wet implementation)

import struct, sys

def write_string(stream, string):
    stream.write(len(string).to_bytes(2, byteorder='big', signed=False))
    for c in string:
        stream.write(ord(c).to_bytes(1, byteorder='big', signed=False))

def register_parser(id, clazz):
    global _parsers

    _parsers[id] = clazz

def create_simple_nbt_class(tag_id, class_tag_name, tag_width, tag_parser):

    class DataNBTTag:

        clazz_width = tag_width
        clazz_name = class_tag_name
        clazz_parser = tag_parser
        clazz_id = tag_id

        @classmethod
        def parse(cls, stream, name):
            return cls(
                tag_value=struct.unpack(
                    cls.clazz_parser, 
                    stream.read(cls.clazz_width)
                )[0],
                tag_name=name
            )

        def __init__(self, tag_value, tag_name='None'):
            int(tag_value)
            self.tag_name = tag_name
            self.tag_value = tag_value

        def print(self, indent=''):
            print(indent + self.__repr__())

        def get(self):
            return self.tag_value

        def name(self):
            return self.tag_name

        def serialize(self, stream, include_name=True):
            if include_name:
                stream.write(type(self).clazz_id.to_bytes(1, byteorder='big', signed=False))
                write_string(stream, self.tag_name)

            stream.write(struct.pack(type(self).clazz_parser, self.tag_value))

        def clone(self):
            return type(self)(self.tag_value, tag_name=self.tag_name)

        def __repr__(self):
            return f'{type(self).clazz_name}Tag \'{self.tag_name}\' = {str(self.tag_value)}'

        def __eq__(self, other):
            return self.tag_name == other.tag_name and self.tag_value == other.tag_value

    register_parser(tag_id, DataNBTTag)

    return DataNBTTag

# def create_string_nbt_class(tag_id):
    class DataNBTTag:

        clazz_id = tag_id

        @classmethod
        def parse(cls, stream, name):
            payload_length = int.from_bytes(stream.read(2), byteorder='big', signed=False)
            payload = stream.read(payload_length).decode('utf-8')
            return cls(payload, tag_name=name)

        def __init__(self, tag_value, tag_name='None'):
            self.tag_name = tag_name
            self.tag_value = tag_value

        def print(self, indent=''):
            print(indent + 'String: ' + self.tag_name + ' = ' + str(self.tag_value))

        def get(self):
            return self.tag_value

        def name(self):
            return self.tag_name

        def serialize(self, stream, include_name=True):
            if include_name:
                stream.write(type(self).clazz_id.to_bytes(1, byteorder='big', signed=False))
                write_string(stream, self.tag_name)
            
            stream.write(len(self.tag_value).to_bytes(2, byteorder='big',signed=False))
            for c in self.tag_value:
                stream.write(ord(c).to_bytes(1, byteorder='big', signed=False))

        def clone(self):
            return type(self)(self.tag_value, tag_name=self.tag_name)

        def __repr__(self):
            return f'StringTag: {self.tag_name} = \'{self.tag_value}\''

        def __eq__(self, other):
            return self.tag_name == other.tag_name and self.tag_value == other.tag_value

    register_parser(tag_id, DataNBTTag)

    return DataNBTTag

def create_array_nbt_class(tag_id, tag_name, sub_type):
    class ArrayNBTTag:

        clazz_sub_type = sub_type
        clazz_name = tag_name
        clazz_id = tag_id

        @classmethod
        def parse(cls, stream, name):
            payload_length = int.from_bytes(stream.read(4), byteorder='big', signed=True)
            tag = cls(tag_name=name)
            for i in range(payload_length):
                tag.add_child(cls.clazz_sub_type.parse(stream, 'None'))
            return tag

        def __init__(self, tag_name='None', children=[]):
            self.tag_name = tag_name
            self.children = children[:]
        
        def add_child(self, tag):
            self.children.append(tag)

        def name(self):
            return self.tag_name

        def print(self, indent=''):
            str_dat = ', '.join([str(c.get()) for c in self.children])
            print(f'{indent}{type(self).clazz_name}: {self.tag_name} size {str(len(self.children))} = [{str_dat}]')

        def get(self):
            return [int(c.get()) for c in self.children]

        def serialize(self, stream, include_name=True):
            if include_name:
                stream.write(type(self).clazz_id.to_bytes(1, byteorder='big', signed=False))
                write_string(stream, self.tag_name)
                
            stream.write(len(self.children).to_bytes(4, byteorder='big', signed=True))

            for tag in self.children:
                tag.serialize(stream, include_name=False)

        def clone(self):
            return type(self)(tag_name=self.tag_name, children=[c.clone() for c in self.children])

        def __repr__(self):
            str_dat = ', '.join([str(c.get()) for c in self.children])
            return f'{type(self).clazz_name}: {self.tag_name} size {str(len(self.children))} = [{str_dat}]'

        def __eq__(self, other):
            return self.tag_name == other.tag_name and \
                len(self.children) == len(other.children) and \
                not any([not self.children[i] == other.children[i] for i in range(len(self.children))])

    register_parser(tag_id, ArrayNBTTag)

    return ArrayNBTTag

def create_list_nbt_class(tag_id):
    class ListNBTTag:

        clazz_id = tag_id

        @classmethod
        def parse(cls, stream, name):
            global _parsers

            sub_type = int.from_bytes(stream.read(1), byteorder='big', signed=False)
            payload_length = int.from_bytes(stream.read(4), byteorder='big', signed=True)
            tag = cls(sub_type, tag_name=name)
            for i in range(payload_length):
                tag.add_child(_parsers[sub_type].parse(stream, 'None'))
            return tag

        def __init__(self, sub_type_id, tag_name='None', children=[]):
            self.tag_name = tag_name
            self.sub_type_id = sub_type_id
            self.children = children[:]
        
        def add_child(self, tag):
            self.children.append(tag)

        def get(self):
            return [c.get() for c in self.children]

        def name(self):
            return self.tag_name

        def print(self, indent=''):
            print(indent + 'List: ' + self.tag_name + ' size ' + str(len(self.children)))
            for c in self.children:
                c.print(indent + '  ')
        
        def serialize(self, stream, include_name=True):
            if include_name:
                stream.write(type(self).clazz_id.to_bytes(1, byteorder='big', signed=False))
                write_string(stream, self.tag_name)
            
            stream.write(self.sub_type_id.to_bytes(1, byteorder='big', signed=False))
            stream.write(len(self.children).to_bytes(4, byteorder='big', signed=True))

            for tag in self.children:
                tag.serialize(stream, include_name=False)

        def clone(self):
            return type(self)(self.sub_type_id, tag_name=self.tag_name, children=[c.clone() for c in self.children])

        def __repr__(self):
            str_dat = ', '.join([c.__repr__() for c in self.children])
            return f'ListTag: {self.tag_name} size {str(len(self.children))} = [{str_dat}]'

        def __eq__(self, other):
            return self.tag_name == other.tag_name and \
                len(self.children) == len(other.children) and \
                (len(self.children) == 0 or not any([not self.children[i] == other.children[i] for i in range(len(self.children))]))

    register_parser(tag_id, ListNBTTag)

    return ListNBTTag

def create_compund_nbt_class(tag_id):
    class CompundNBTTag:

        clazz_id = tag_id

        @classmethod
        def parse(cls, stream, name):
            tag = cls(tag_name=name)
            while stream.peek() != 0: # end tag
                tag.add_child(parse_nbt(stream))
            stream.read(1) # get rid of the end tag
            return tag

        def __init__(self, tag_name='None', children=[]):
            self.tag_name = tag_name
            self.children = { c.tag_name: c for c in children[:] }
        
        def add_child(self, tag):
            self.children[tag.tag_name] = tag

        def get(self, name):
            return self.children[name]

        # def get(self):
        #     return { n: v.get() for n, v in self.children }

        def name(self):
            return self.tag_name

        def has(self, name):
            return name in self.children

        def to_dict(self):
            nd = {}
            for p in self.children:
                nd[p] = self.children[p].get()
            return nd

        def print(self, indent=''):
            print(indent + 'Compound: ' + self.tag_name + ' size ' + str(len(self.children)))
            for c in self.children:
                self.children[c].print(indent + '  ')

        def serialize(self, stream, include_name=True):
            if include_name:
                stream.write(type(self).clazz_id.to_bytes(1, byteorder='big', signed=False))
                write_string(stream, self.tag_name)
            
            for tag_name in self.children:
                self.children[tag_name].serialize(stream, include_name=True)
            
            stream.write((0).to_bytes(1, byteorder='big', signed=False))

        def clone(self):
            return type(self)(tag_name=self.tag_name, children=[v.clone() for k, v in self.children.items()])

        def __repr__(self):
            str_dat = ', '.join([c.__repr__() for name, c in self.children.items()])
            return f'CompundTag: {self.tag_name} size {str(len(self.children))} = {{{str_dat}}}]'

        def __eq__(self, other):
            passed = True
            for name, v in self.children.items():
                if name not in other.children:
                    passed = False
                elif other.children[name] != v:
                    passed = False
            return self.tag_name == other.tag_name and \
                len(self.children) == len(other.children) and \
                passed

    register_parser(tag_id, CompundNBTTag)

    return CompundNBTTag

_parsers = {}


# %%

# ByteTag = create_simple_nbt_class(1, 'Byte', 1, '>b')

class DataNBTTag1:

    tag_id, class_tag_name, tag_width, tag_parser = 1, 'Byte', 1, '>b'
    clazz_width = tag_width
    clazz_name = class_tag_name
    clazz_parser = tag_parser
    clazz_id = tag_id

    @classmethod
    def parse(cls, stream, name):
        return cls(
            tag_value=struct.unpack(
                cls.clazz_parser, 
                stream.read(cls.clazz_width)
            )[0],
            tag_name=name
        )

    def __init__(self, tag_value, tag_name='None'):
        int(tag_value)
        self.tag_name = tag_name
        self.tag_value = tag_value

    def print(self, indent=''):
        print(indent + self.__repr__())

    def get(self):
        return self.tag_value

    def name(self):
        return self.tag_name

    def serialize(self, stream, include_name=True):
        if include_name:
            stream.write(type(self).clazz_id.to_bytes(1, byteorder='big', signed=False))
            write_string(stream, self.tag_name)

        stream.write(struct.pack(type(self).clazz_parser, self.tag_value))

    def clone(self):
        return type(self)(self.tag_value, tag_name=self.tag_name)

    def __repr__(self):
        return f'{type(self).clazz_name}Tag \'{self.tag_name}\' = {str(self.tag_value)}'

    def __eq__(self, other):
        return self.tag_name == other.tag_name and self.tag_value == other.tag_value

register_parser(1, DataNBTTag1)
ByteTag = DataNBTTag1

# %%

# ShortTag = create_simple_nbt_class(2, 'Short', 2, '>h')

class DataNBTTag2:

    tag_id, class_tag_name, tag_width, tag_parser = 2, 'Short', 2, '>h'
    clazz_width = tag_width
    clazz_name = class_tag_name
    clazz_parser = tag_parser
    clazz_id = tag_id

    @classmethod
    def parse(cls, stream, name):
        return cls(
            tag_value=struct.unpack(
                cls.clazz_parser, 
                stream.read(cls.clazz_width)
            )[0],
            tag_name=name
        )

    def __init__(self, tag_value, tag_name='None'):
        int(tag_value)
        self.tag_name = tag_name
        self.tag_value = tag_value

    def print(self, indent=''):
        print(indent + self.__repr__())

    def get(self):
        return self.tag_value

    def name(self):
        return self.tag_name

    def serialize(self, stream, include_name=True):
        if include_name:
            stream.write(type(self).clazz_id.to_bytes(1, byteorder='big', signed=False))
            write_string(stream, self.tag_name)

        stream.write(struct.pack(type(self).clazz_parser, self.tag_value))

    def clone(self):
        return type(self)(self.tag_value, tag_name=self.tag_name)

    def __repr__(self):
        return f'{type(self).clazz_name}Tag \'{self.tag_name}\' = {str(self.tag_value)}'

    def __eq__(self, other):
        return self.tag_name == other.tag_name and self.tag_value == other.tag_value

register_parser(2, DataNBTTag2)
ShortTag = DataNBTTag2

# %%

# IntTag = create_simple_nbt_class(3, 'Int', 4, '>i')

class DataNBTTag3:

    tag_id, class_tag_name, tag_width, tag_parser = 3, 'Int', 4, '>i'
    clazz_width = tag_width
    clazz_name = class_tag_name
    clazz_parser = tag_parser
    clazz_id = tag_id

    @classmethod
    def parse(cls, stream, name):
        return cls(
            tag_value=struct.unpack(
                cls.clazz_parser, 
                stream.read(cls.clazz_width)
            )[0],
            tag_name=name
        )

    def __init__(self, tag_value, tag_name='None'):
        int(tag_value)
        self.tag_name = tag_name
        self.tag_value = tag_value

    def print(self, indent=''):
        print(indent + self.__repr__())

    def get(self):
        return self.tag_value

    def name(self):
        return self.tag_name

    def serialize(self, stream, include_name=True):
        if include_name:
            stream.write(type(self).clazz_id.to_bytes(1, byteorder='big', signed=False))
            write_string(stream, self.tag_name)

        stream.write(struct.pack(type(self).clazz_parser, self.tag_value))

    def clone(self):
        return type(self)(self.tag_value, tag_name=self.tag_name)

    def __repr__(self):
        return f'{type(self).clazz_name}Tag \'{self.tag_name}\' = {str(self.tag_value)}'

    def __eq__(self, other):
        return self.tag_name == other.tag_name and self.tag_value == other.tag_value

register_parser(3, DataNBTTag3)
IntTag = DataNBTTag3

# %%

# LongTag = create_simple_nbt_class(4, 'Long', 8, '>q')

class DataNBTTag4:

    tag_id, class_tag_name, tag_width, tag_parser = 4, 'Long', 8, '>q'
    clazz_width = tag_width
    clazz_name = class_tag_name
    clazz_parser = tag_parser
    clazz_id = tag_id

    @classmethod
    def parse(cls, stream, name):
        return cls(
            tag_value=struct.unpack(
                cls.clazz_parser, 
                stream.read(cls.clazz_width)
            )[0],
            tag_name=name
        )

    def __init__(self, tag_value, tag_name='None'):
        int(tag_value)
        self.tag_name = tag_name
        self.tag_value = tag_value

    def print(self, indent=''):
        print(indent + self.__repr__())

    def get(self):
        return self.tag_value

    def name(self):
        return self.tag_name

    def serialize(self, stream, include_name=True):
        if include_name:
            stream.write(type(self).clazz_id.to_bytes(1, byteorder='big', signed=False))
            write_string(stream, self.tag_name)

        stream.write(struct.pack(type(self).clazz_parser, self.tag_value))

    def clone(self):
        return type(self)(self.tag_value, tag_name=self.tag_name)

    def __repr__(self):
        return f'{type(self).clazz_name}Tag \'{self.tag_name}\' = {str(self.tag_value)}'

    def __eq__(self, other):
        return self.tag_name == other.tag_name and self.tag_value == other.tag_value

register_parser(4, DataNBTTag4)
LongTag = DataNBTTag4

# %%

# FloatTag = create_simple_nbt_class(5, 'Float', 4, '>f')

class DataNBTTag5:

    tag_id, class_tag_name, tag_width, tag_parser = 5, 'Float', 4, '>f'
    clazz_width = tag_width
    clazz_name = class_tag_name
    clazz_parser = tag_parser
    clazz_id = tag_id

    @classmethod
    def parse(cls, stream, name):
        return cls(
            tag_value=struct.unpack(
                cls.clazz_parser, 
                stream.read(cls.clazz_width)
            )[0],
            tag_name=name
        )

    def __init__(self, tag_value, tag_name='None'):
        int(tag_value)
        self.tag_name = tag_name
        self.tag_value = tag_value

    def print(self, indent=''):
        print(indent + self.__repr__())

    def get(self):
        return self.tag_value

    def name(self):
        return self.tag_name

    def serialize(self, stream, include_name=True):
        if include_name:
            stream.write(type(self).clazz_id.to_bytes(1, byteorder='big', signed=False))
            write_string(stream, self.tag_name)

        stream.write(struct.pack(type(self).clazz_parser, self.tag_value))

    def clone(self):
        return type(self)(self.tag_value, tag_name=self.tag_name)

    def __repr__(self):
        return f'{type(self).clazz_name}Tag \'{self.tag_name}\' = {str(self.tag_value)}'

    def __eq__(self, other):
        return self.tag_name == other.tag_name and self.tag_value == other.tag_value

register_parser(5, DataNBTTag5)
FloatTag = DataNBTTag5

# %%

# DoubleTag = create_simple_nbt_class(6, 'Double', 8, '>d')

class DataNBTTag6:

    tag_id, class_tag_name, tag_width, tag_parser = 6, 'Double', 8, '>d'
    clazz_width = tag_width
    clazz_name = class_tag_name
    clazz_parser = tag_parser
    clazz_id = tag_id

    @classmethod
    def parse(cls, stream, name):
        return cls(
            tag_value=struct.unpack(
                cls.clazz_parser, 
                stream.read(cls.clazz_width)
            )[0],
            tag_name=name
        )

    def __init__(self, tag_value, tag_name='None'):
        int(tag_value)
        self.tag_name = tag_name
        self.tag_value = tag_value

    def print(self, indent=''):
        print(indent + self.__repr__())

    def get(self):
        return self.tag_value

    def name(self):
        return self.tag_name

    def serialize(self, stream, include_name=True):
        if include_name:
            stream.write(type(self).clazz_id.to_bytes(1, byteorder='big', signed=False))
            write_string(stream, self.tag_name)

        stream.write(struct.pack(type(self).clazz_parser, self.tag_value))

    def clone(self):
        return type(self)(self.tag_value, tag_name=self.tag_name)

    def __repr__(self):
        return f'{type(self).clazz_name}Tag \'{self.tag_name}\' = {str(self.tag_value)}'

    def __eq__(self, other):
        return self.tag_name == other.tag_name and self.tag_value == other.tag_value

register_parser(6, DataNBTTag6)
DoubleTag = DataNBTTag6

# %%

# ByteArrayTag = create_array_nbt_class(7, 'ByteArray', ByteTag)
class ArrayNBTTag1:
    tag_id, tag_name, sub_type = 7, 'ByteArray', ByteTag
    clazz_sub_type = sub_type
    clazz_name = tag_name
    clazz_id = tag_id

    @classmethod
    def parse(cls, stream, name):
        payload_length = int.from_bytes(stream.read(4), byteorder='big', signed=True)
        tag = cls(tag_name=name)
        for i in range(payload_length):
            tag.add_child(cls.clazz_sub_type.parse(stream, 'None'))
        return tag

    def __init__(self, tag_name='None', children=[]):
        self.tag_name = tag_name
        self.children = children[:]
    
    def add_child(self, tag):
        self.children.append(tag)

    def name(self):
        return self.tag_name

    def print(self, indent=''):
        str_dat = ', '.join([str(c.get()) for c in self.children])
        print(f'{indent}{type(self).clazz_name}: {self.tag_name} size {str(len(self.children))} = [{str_dat}]')

    def get(self):
        return [int(c.get()) for c in self.children]

    def serialize(self, stream, include_name=True):
        if include_name:
            stream.write(type(self).clazz_id.to_bytes(1, byteorder='big', signed=False))
            write_string(stream, self.tag_name)
            
        stream.write(len(self.children).to_bytes(4, byteorder='big', signed=True))

        for tag in self.children:
            tag.serialize(stream, include_name=False)

    def clone(self):
        return type(self)(tag_name=self.tag_name, children=[c.clone() for c in self.children])

    def __repr__(self):
        str_dat = ', '.join([str(c.get()) for c in self.children])
        return f'{type(self).clazz_name}: {self.tag_name} size {str(len(self.children))} = [{str_dat}]'

    def __eq__(self, other):
        return self.tag_name == other.tag_name and \
            len(self.children) == len(other.children) and \
            not any([not self.children[i] == other.children[i] for i in range(len(self.children))])\

ByteArrayTag = ArrayNBTTag1
register_parser(7, ArrayNBTTag1)

# %%

# IntArrayTag = create_array_nbt_class(11, 'IntArray', IntTag)
class ArrayNBTTag2:
    tag_id, tag_name, sub_type = 11, 'IntArray', IntTag
    clazz_sub_type = sub_type
    clazz_name = tag_name
    clazz_id = tag_id

    @classmethod
    def parse(cls, stream, name):
        payload_length = int.from_bytes(stream.read(4), byteorder='big', signed=True)
        tag = cls(tag_name=name)
        for i in range(payload_length):
            tag.add_child(cls.clazz_sub_type.parse(stream, 'None'))
        return tag

    def __init__(self, tag_name='None', children=[]):
        self.tag_name = tag_name
        self.children = children[:]
    
    def add_child(self, tag):
        self.children.append(tag)

    def name(self):
        return self.tag_name

    def print(self, indent=''):
        str_dat = ', '.join([str(c.get()) for c in self.children])
        print(f'{indent}{type(self).clazz_name}: {self.tag_name} size {str(len(self.children))} = [{str_dat}]')

    def get(self):
        return [int(c.get()) for c in self.children]

    def serialize(self, stream, include_name=True):
        if include_name:
            stream.write(type(self).clazz_id.to_bytes(1, byteorder='big', signed=False))
            write_string(stream, self.tag_name)
            
        stream.write(len(self.children).to_bytes(4, byteorder='big', signed=True))

        for tag in self.children:
            tag.serialize(stream, include_name=False)

    def clone(self):
        return type(self)(tag_name=self.tag_name, children=[c.clone() for c in self.children])

    def __repr__(self):
        str_dat = ', '.join([str(c.get()) for c in self.children])
        return f'{type(self).clazz_name}: {self.tag_name} size {str(len(self.children))} = [{str_dat}]'

    def __eq__(self, other):
        return self.tag_name == other.tag_name and \
            len(self.children) == len(other.children) and \
            not any([not self.children[i] == other.children[i] for i in range(len(self.children))])\

IntArrayTag = ArrayNBTTag2
register_parser(11, ArrayNBTTag2)

# %%

# LongArrayTag = create_array_nbt_class(12, 'LongArray', LongTag)
class ArrayNBTTag3:
    tag_id, tag_name, sub_type = 12, 'LongArray', LongTag
    clazz_sub_type = sub_type
    clazz_name = tag_name
    clazz_id = tag_id

    @classmethod
    def parse(cls, stream, name):
        payload_length = int.from_bytes(stream.read(4), byteorder='big', signed=True)
        tag = cls(tag_name=name)
        for i in range(payload_length):
            tag.add_child(cls.clazz_sub_type.parse(stream, 'None'))
        return tag

    def __init__(self, tag_name='None', children=[]):
        self.tag_name = tag_name
        self.children = children[:]
    
    def add_child(self, tag):
        self.children.append(tag)

    def name(self):
        return self.tag_name

    def print(self, indent=''):
        str_dat = ', '.join([str(c.get()) for c in self.children])
        print(f'{indent}{type(self).clazz_name}: {self.tag_name} size {str(len(self.children))} = [{str_dat}]')

    def get(self):
        return [int(c.get()) for c in self.children]

    def serialize(self, stream, include_name=True):
        if include_name:
            stream.write(type(self).clazz_id.to_bytes(1, byteorder='big', signed=False))
            write_string(stream, self.tag_name)
            
        stream.write(len(self.children).to_bytes(4, byteorder='big', signed=True))

        for tag in self.children:
            tag.serialize(stream, include_name=False)

    def clone(self):
        return type(self)(tag_name=self.tag_name, children=[c.clone() for c in self.children])

    def __repr__(self):
        str_dat = ', '.join([str(c.get()) for c in self.children])
        return f'{type(self).clazz_name}: {self.tag_name} size {str(len(self.children))} = [{str_dat}]'

    def __eq__(self, other):
        return self.tag_name == other.tag_name and \
            len(self.children) == len(other.children) and \
            not any([not self.children[i] == other.children[i] for i in range(len(self.children))])\

LongArrayTag = ArrayNBTTag3
register_parser(12, ArrayNBTTag3)

# %%

# StringTag = create_string_nbt_class(8)
class DataNBTTag7:

    clazz_id = 8

    @classmethod
    def parse(cls, stream, name):
        payload_length = int.from_bytes(stream.read(2), byteorder='big', signed=False)
        payload = stream.read(payload_length).decode('utf-8')
        return cls(payload, tag_name=name)

    def __init__(self, tag_value, tag_name='None'):
        self.tag_name = tag_name
        self.tag_value = tag_value

    def print(self, indent=''):
        print(indent + 'String: ' + self.tag_name + ' = ' + str(self.tag_value))

    def get(self):
        return self.tag_value

    def name(self):
        return self.tag_name

    def serialize(self, stream, include_name=True):
        if include_name:
            stream.write(type(self).clazz_id.to_bytes(1, byteorder='big', signed=False))
            write_string(stream, self.tag_name)
        
        stream.write(len(self.tag_value).to_bytes(2, byteorder='big',signed=False))
        for c in self.tag_value:
            stream.write(ord(c).to_bytes(1, byteorder='big', signed=False))

    def clone(self):
        return type(self)(self.tag_value, tag_name=self.tag_name)

    def __repr__(self):
        return f'StringTag: {self.tag_name} = \'{self.tag_value}\''

    def __eq__(self, other):
        return self.tag_name == other.tag_name and self.tag_value == other.tag_value

register_parser(8, DataNBTTag7)
StringTag = DataNBTTag7

# %%

# ListTag = create_list_nbt_class(9)
class ListNBTTag:

    clazz_id = 9

    @classmethod
    def parse(cls, stream, name):
        global _parsers

        sub_type = int.from_bytes(stream.read(1), byteorder='big', signed=False)
        payload_length = int.from_bytes(stream.read(4), byteorder='big', signed=True)
        tag = cls(sub_type, tag_name=name)
        for i in range(payload_length):
            tag.add_child(_parsers[sub_type].parse(stream, 'None'))
        return tag

    def __init__(self, sub_type_id, tag_name='None', children=[]):
        self.tag_name = tag_name
        self.sub_type_id = sub_type_id
        self.children = children[:]
    
    def add_child(self, tag):
        self.children.append(tag)

    def get(self):
        return [c.get() for c in self.children]

    def name(self):
        return self.tag_name

    def print(self, indent=''):
        print(indent + 'List: ' + self.tag_name + ' size ' + str(len(self.children)))
        for c in self.children:
            c.print(indent + '  ')
    
    def serialize(self, stream, include_name=True):
        if include_name:
            stream.write(type(self).clazz_id.to_bytes(1, byteorder='big', signed=False))
            write_string(stream, self.tag_name)
        
        stream.write(self.sub_type_id.to_bytes(1, byteorder='big', signed=False))
        stream.write(len(self.children).to_bytes(4, byteorder='big', signed=True))

        for tag in self.children:
            tag.serialize(stream, include_name=False)

    def clone(self):
        return type(self)(self.sub_type_id, tag_name=self.tag_name, children=[c.clone() for c in self.children])

    def __repr__(self):
        str_dat = ', '.join([c.__repr__() for c in self.children])
        return f'ListTag: {self.tag_name} size {str(len(self.children))} = [{str_dat}]'

    def __eq__(self, other):
        return self.tag_name == other.tag_name and \
            len(self.children) == len(other.children) and \
            (len(self.children) == 0 or not any([not self.children[i] == other.children[i] for i in range(len(self.children))]))

register_parser(9, ListNBTTag)
ListTag = ListNBTTag

# %%

# CompoundTag = create_compund_nbt_class(10)
class CompundNBTTag:

    clazz_id = 10

    @classmethod
    def parse(cls, stream, name):
        tag = cls(tag_name=name)
        while stream.peek() != 0: # end tag
            tag.add_child(parse_nbt(stream))
        stream.read(1) # get rid of the end tag
        return tag

    def __init__(self, tag_name='None', children=[]):
        self.tag_name = tag_name
        self.children = { c.tag_name: c for c in children[:] }
    
    def add_child(self, tag):
        self.children[tag.tag_name] = tag

    def get(self, name):
        return self.children[name]

    # def get(self):
    #     return { n: v.get() for n, v in self.children }

    def name(self):
        return self.tag_name

    def has(self, name):
        return name in self.children

    def to_dict(self):
        nd = {}
        for p in self.children:
            nd[p] = self.children[p].get()
        return nd

    def print(self, indent=''):
        print(indent + 'Compound: ' + self.tag_name + ' size ' + str(len(self.children)))
        for c in self.children:
            self.children[c].print(indent + '  ')

    def serialize(self, stream, include_name=True):
        if include_name:
            stream.write(type(self).clazz_id.to_bytes(1, byteorder='big', signed=False))
            write_string(stream, self.tag_name)
        
        for tag_name in self.children:
            self.children[tag_name].serialize(stream, include_name=True)
        
        stream.write((0).to_bytes(1, byteorder='big', signed=False))

    def clone(self):
        return type(self)(tag_name=self.tag_name, children=[v.clone() for k, v in self.children.items()])

    def __repr__(self):
        str_dat = ', '.join([c.__repr__() for name, c in self.children.items()])
        return f'CompundTag: {self.tag_name} size {str(len(self.children))} = {{{str_dat}}}]'

    def __eq__(self, other):
        passed = True
        for name, v in self.children.items():
            if name not in other.children:
                passed = False
            elif other.children[name] != v:
                passed = False
        return self.tag_name == other.tag_name and \
            len(self.children) == len(other.children) and \
            passed

CompoundTag = CompundNBTTag
register_parser(10, CompundNBTTag)

# %%

def parse_nbt(stream):
    global _parsers

    tag_type = int.from_bytes(stream.read(1), byteorder='big', signed=False)
    tag_name_length = int.from_bytes(stream.read(2), byteorder='big', signed=False)
    tag_name = stream.read(tag_name_length).decode('utf-8')

    return _parsers[tag_type].parse(stream, tag_name)

