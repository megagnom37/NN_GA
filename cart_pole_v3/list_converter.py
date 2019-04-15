import numpy as np

def to_list(ls, values_type):
  res = []
  if not isinstance(ls, values_type):
    for i in range(len(ls)):
      res.extend(to_list(ls[i], values_type))
  else:
    res.append(ls)

  return res


def recover_list(ls, ls2, values_type):
  if not isinstance(ls, values_type):
    for i in range(len(ls)):
      if isinstance(ls[i], values_type):
        ls[i] = ls2[0]
        ls2.pop(0)
      else:
        recover_list(ls[i], ls2, values_type)
  else:
    return ls