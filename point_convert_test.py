from find_point import point_convert

four_point = [[10, 10], [870, 10], [870, 550], [10, 550]]

x, y = 10, 10
print(x, y, point_convert((x, y), four_point))

x, y = 440, 280
print(x, y, point_convert((x, y), four_point))

x, y = 870, 550
print(x, y, point_convert((x, y), four_point))
