# https://github.com/Angtian/BboxTools/blob/master/BboxTools.py (May 13, 2024)

# coding=utf-8
# Copyright (c) 2019-2021 Angtian Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import warnings

try:
    import cv2

    enable_vc2 = True
    resize_method = cv2.INTER_AREA
except:
    enable_vc2 = False

try:
    from PIL import Image
    from PIL import ImageFont
    from PIL import ImageDraw

    enable_PIL = True
except:
    enable_PIL = False

set_enable_pytorch = True

if set_enable_pytorch:
    try:
        import torch

        enable_pytorch = True

        resize_method_torch = 'bilinear'
    except:
        enable_pytorch = False
else:
    enable_pytorch = False


class Bbox2D(object):
    bbox = None
    boundary = None

    def __init__(self, bbox, image_boundary=None):
        self.set_bbox(bbox, check=False)
        if isinstance(image_boundary, np.ndarray):
            image_boundary = image_boundary.tolist()
        if not self.bbox:
            return
        if image_boundary:
            self.set_boundary(list(image_boundary))
        if not self.check_box(bbox=self.bbox, img_shape=self.boundary):
            self.illegal_bbox_exception()
        object.__setattr__(self, 'attributes', {})

    def __len__(self):
        if not self.bbox:
            return 0
        return 2

    def __getitem__(self, item):
        if self.if_empty():
            raise Exception('Cannot get item from empty box!')
        if item == 'x' or item == 0:
            return tuple(self.bbox[0])
        elif item == 'y' or item == 1:
            return tuple(self.bbox[1])
        else:
            if type(item) == int:
                raise Exception(' Input index out of range, which should be 0,1 but get: ' + str(item))
            raise Exception('Illegal index!')

    def __eq__(self, other):
        if self.if_empty() or other.if_empty():
            return self.if_empty() == other.if_empty()

        for i in range(2):
            if not self.bbox[i][0] == other.bbox[i][0]:
                return False
            if not self.bbox[i][1] == other.bbox[i][1]:
                return False
        return True

    def __or__(self, other):
        if self.if_empty():
            return self.copy()
        if other.if_empty():
            return other.copy()

        tem = []
        for i in range(2):
            tem.append((min(self.bbox[i][0], other.bbox[i][0]), max(self.bbox[i][1], other.bbox[i][1])))
        equal = False
        if self.boundary and other.boundary:
            equal = True
            for i in range(2):
                if not self.boundary[i] == other.boundary[i]:
                    equal = False
        if equal:
            boundary = self.boundary
        else:
            boundary = None

        return Bbox2D(tem, boundary)

    def __and__(self, other):
        if self.if_empty():
            return self.copy()
        if other.if_empty():
            return other.copy()

        tem = []
        for i in range(2):
            tem.append((max(self.bbox[i][0], other.bbox[i][0]), min(self.bbox[i][1], other.bbox[i][1])))
        equal = False
        if self.boundary and other.boundary:
            equal = True
            for i in range(2):
                if not self.boundary[i] == other.boundary[i]:
                    equal = False
        if equal:
            if not self.check_box(tem, self.boundary):
                return Bbox2D(None)
        else:
            if not self.check_box(tem):
                return Bbox2D(None)

        if equal:
            boundary = self.boundary
        else:
            boundary = None

        return Bbox2D(tem, boundary)

    def __mul__(self, other):
        if self.if_empty():
            return empty()
        if type(other) == int or type(other) == float:
            if other < 0:
                raise Exception('Input must be positive! Got ' + str(other))
            self_ = self.copy()
            # new_shape = [int(tem * other) for tem in self.shape]
            # self.set_shape(new_shape, center=self.center)
            self_.bbox[0] = (int(self_.bbox[0][0] * other), int(self_.bbox[0][1] * other))
            self_.bbox[1] = (int(self_.bbox[1][0] * other), int(self_.bbox[1][1] * other))

            if self_.boundary:
                self_.boundary = (int(self_.boundary[0] * other), int(self_.boundary[1] * other))
            return self_
        elif (type(other) == tuple or type(other) == list) and len(other) == 2:
            self_ = self.copy()
            for i, tem in enumerate(other):
                if type(tem) != int and type(tem) != float:
                    raise Exception('Input must be int or float, got %s' % str(other))
                if tem < 0:
                    raise Exception('Input must be positive! Got ' + str(other))
                self_.bbox[i] = (int(self_.bbox[i][0] * tem), int(self_.bbox[i][1] * tem))
            if self_.boundary:
                self_.boundary = (int(self_.boundary[0] * other[0]), int(self_.boundary[1] * other[1]))
            return self_
        else:
            raise Exception('Multiply method only support int or float input, got %s' % str(other))

    def __copy__(self):
        if self.if_empty():
            return empty()
        out = Bbox2D(self.bbox, self.boundary)
        for attr in self.attributes.keys():
            out.__setattr__(attr, self.attributes[attr])
        return out

    def __str__(self):
        if self.bbox is None:
            return '<class "Bbox2D", Empty box>'
        if not self.boundary:
            out = '<class "Bbox2D", bbox=[(%d, %d), (%d, %d)]' % (
            self.bbox[0][0], self.bbox[0][1], self.bbox[1][0], self.bbox[1][1])
        else:
            out = '<class "Bbox2D", bbox=[(%d, %d), (%d, %d)], boundary=[%d, %d]' % (
            self.bbox[0][0], self.bbox[0][1], self.bbox[1][0], self.bbox[1][1], self.boundary[0], self.boundary[1])
        for k in self.attributes.keys():
            if not k == 'attributes':
                out += ', ' + k + '=' + str(self.attributes[k])
        out += '>'
        return out

    def __repr__(self):
        return self.__str__()

    def __bool__(self):
        if not self.if_empty():
            return True
        else:
            return False

    def __setattr__(self, key, value):
        if key == 'bbox':
            object.__setattr__(self, key, value)
            return
        if key == 'boundary':
            object.__setattr__(self, key, value)
            return
        if type(key) == int:
            raise Exception('In order to avoid misunderstanding, setitem method does not accept int as attribute index.'
                            'Please use one_point method to set the shape of bbox.')
        if not type(key) == str:
            raise Exception('In order to avoid misunderstanding, keys of box attributes can only be str.')
        self.attributes[key] = value

    def __getattr__(self, item):
        if not item in list(self.attributes.keys()) + ['bbox', 'boundary']:
            raise Exception('No attributes %s!' % item)
        if item == 'bbox':
            return self.bbox
        if item == 'boundary':
            return tuple(self.boundary)
        return self.attributes[item]

    @property
    def shape(self):
        if self.if_empty():
            return 0, 0
        return self._shape_along_axis(0), self._shape_along_axis(1)

    @property
    def size(self):
        if self.if_empty():
            return 0
        return self._shape_along_axis(0) * self._shape_along_axis(1)

    @property
    def center(self):
        if self.if_empty():
            return None
        return self._get_center()

    @property
    def lu(self):
        if self.if_empty():
            return None
        return self.bbox[0][0], self.bbox[1][0]

    @property
    def rb(self):
        if self.if_empty():
            return None
        return self.bbox[0][1], self.bbox[1][1]

    @property
    def ru(self):
        if self.if_empty():
            return None
        return self.bbox[0][1], self.bbox[1][0]

    @property
    def lb(self):
        if self.if_empty():
            return None
        return self.bbox[0][0], self.bbox[1][1]

    def if_empty(self):
        """
        Whether this bbox is empty. True for empty bbox.
        :return: (bool)
        """
        return self.bbox is None

    def pillow_bbox(self):
        """
        Get bbox in pillow format
        :return: [x0, y0, x1, y1]
        """
        if self.if_empty():
            return [0, 0, 0, 0]
        return [self.bbox[1][0], self.bbox[0][0], self.bbox[1][1], self.bbox[0][1]]

    def get_four_corner(self):
        return self.lu, self.rb, self.ru, self.lb

    def assign_attr(self, **kwargs):
        for k, v in kwargs.items():
            self.attributes[str(k)] = v

    def copy(self):
        return self.__copy__()

    def include(self, other):
        """
        Check if other is inside this box. Notice include means strictly include, other could not place at the boundary
        of this bbox.
        :param other: (Bbox2D or tuple of int or 2d ndarray with shape (n, 2)) bbox or point(s)
        :return: (bool) True or False
        """
        if type(other) == Bbox2D:
            out = True
            for i in range(2):
                if self.bbox[i][0] > other.bbox[i][0]:
                    out = False
                if self.bbox[i][1] < other.bbox[i][1]:
                    out = False
            return out

        if (type(other) == tuple and len(other) == 2) or (type(other) == np.ndarray and len(other.shape) == 1):
            if other[0] < self.bbox[0][0] or other[0] >= self.bbox[0][1]:
                return False
            if other[1] < self.bbox[1][0] or other[1] >= self.bbox[1][1]:
                return False
            return True

        if type(other) == np.ndarray and len(other.shape) == 2:
            return np.logical_and(np.logical_and(self.bbox[0][0] <= other[:, 0], other[:, 0] < self.bbox[0][1]),
                                  np.logical_and(self.bbox[1][0] <= other[:, 1], other[:, 1] < self.bbox[1][1]))

        raise Exception('Include method suppose to be point or bbox, but got %s' % str(other))

    def exclude(self, other, axis):
        out = self.copy()
        if not self.if_include(other):
            raise Exception('The other box is not inside this box')
        if self.bbox[axis][0] == other.bbox[axis][0]:
            out.one_point(axis, 0, other[axis][1])
        elif self.bbox[axis][1] == other.bbox[axis][1]:
            out.one_point(axis, 1, other[axis][0])
        else:
            raise Exception('Boundary unmatched! Cannot exclude one from the other!')
        if not out.check_box():
            self.illegal_bbox_exception()
        return out

    def set_bbox(self, bbox, check=True):
        if not bbox:
            self.bbox = None
            return
        if type(bbox) == Bbox2D:
            self.bbox = bbox.bbox
            self.boundary = bbox.boundary
        else:
            self.bbox = [[int(tem[0]), int(tem[1])] for tem in bbox]
        if check:
            if not self.check_box(bbox=self.bbox, img_shape=self.boundary):
                self.illegal_bbox_exception()

    def one_point(self, axis, num, value, change=False, auto_correct=True):
        if axis > 1 or axis < 0 or num > 1 or num < 0:
            raise Exception('Axis or num beyond limit!')
        if change:
            self.bbox[axis][num] += value
        else:
            self.bbox[axis][num] = value
        tem = None
        if auto_correct:
            tem = self._limit_to_boundary(self.boundary)
        if not self.check_box(bbox=self.bbox, img_shape=self.boundary):
            self.illegal_bbox_exception()
        return tem

    def set_boundary(self, boundary):
        if len(boundary) == 3:
            boundary = boundary[0:2]
        self.boundary = boundary
        self._limit_to_boundary(boundary)
        return self

    def _shape_along_axis(self, axis):
        return self.bbox[axis][1] - self.bbox[axis][0]

    def _limit_to_boundary(self, boundary):
        if self.check_box(img_shape=boundary):
            return
        output = self
        for i in range(2):
            if boundary[i] < self.bbox[i][1]:
                output = self.bbox[i][1] - boundary[i]
                self.bbox[i] = [self.bbox[i][0], boundary[i]]
            if self.bbox[i][0] < 0:
                output = -self.bbox[i][0]
                self.bbox[i] = [0, self.bbox[i][1]]
        return output

    def check_box(self, bbox=None, img_shape=None):
        if not bbox:
            bbox = self.bbox
        if not img_shape:
            img_shape = self.boundary
        if not len(bbox) == 2:
            return False
        for tem in bbox:
            if not (type(tem) == list or type(tem) == tuple):
                return False
            if not len(tem) == 2:
                return False
            if not (type(tem[0]) == int and type(tem[1]) == int):
                return False
            if tem[1] < tem[0]:
                return False

        if img_shape:
            for i in range(2):
                if img_shape[i] < bbox[i][1]:
                    return False
                if bbox[i][0] < 0:
                    return False

        return True

    def illegal_bbox_exception(self):
        if self.bbox and self.boundary:
            raise Exception('Illegal boundary box with shape [(%d, %d), (%d, %d)] and boundary [%d, %d]' % (
            self.bbox[0][0], self.bbox[0][1], self.bbox[1][0], self.bbox[1][1], self.boundary[0], self.boundary[1]))
        elif self.bbox:
            raise Exception('Illegal boundary box with shape [(%d, %d), (%d, %d)]' % (
            self.bbox[0][0], self.bbox[0][1], self.bbox[1][0], self.bbox[1][1]))
        else:
            raise Exception('Empty boundary box!')

    def apply(self, image, copy=False, allow_padding=True):
        """
        Crop image by this bbox. The output size would be: h_out, w_out = self.size.
        Examples:
        >>> a = np.arange(9).reshape((3, 3))
        >>> a
        array([[0, 1, 2],
               [3, 4, 5],
               [6, 7, 8]])
        >>> bbt.Bbox2D([(0, 2), (1, 3)]).apply(a)
        array([[1, 2],
               [4, 5]])
        :param image: (np.ndarray or torch.Tensor) Source image. ndarray should have shape (h, w) or (h, w, c)
                         Tensor should have shape (h, w) or (c, h, w) or (n, c, h, w)
        :param copy: (bool) Whether copy the cropped image
        :param allow_padding: (bool) Whether pad the image when the crop box is partially outside the image,
                         Effective only if self.boundary is None.
        :return: (np.ndarray or torch.Tensor) crop image
        """
        if type(image) == np.ndarray:
            type_ = 'numpy'
        elif enable_pytorch and type(image) == torch.Tensor:
            type_ = 'torch'
        else:
            raise Exception('Image must be either np.ndarray or torch.Tensor, got %s.' % str(type(image)))

        _check_image_size(image, self.boundary)

        if type_ == 'numpy':
            if allow_padding and self.boundary is None:
                if copy:
                    return _apply_pad_numpy(image, self).copy()
                return _apply_pad_numpy(image, self)
            else:
                if copy:
                    return _apply_bbox_numpy(image, self.bbox).copy()
                return _apply_bbox_numpy(image, self.bbox)

        if type_ == 'torch':
            if allow_padding and self.boundary is None:
                if copy:
                    return _apply_pad_torch(image, self).contiguous().clone()
                return _apply_pad_torch(image, self)
            else:
                if copy:
                    return _apply_bbox_torch(image, self.bbox).contiguous().clone()
                return _apply_bbox_torch(image, self.bbox)

    def assign(self, image, value, auto_fit=True):
        """
        Fill in-box-area of the image with given value. Notice instead of checking whether the bbox is out of boundary
        of the image when boundary of this bbox is None, this function will temporarily set the bbox to be limited
        inside the image boundary, which might cause a Unpaired shape error when auto_fit is disabled, or might have
        unexpected manner when auto_fit is enabled. Thus, a bbox with boundary is strongly suggested.
        Examples:
        >>> a = np.zeros((3, 3))
        >>> a
        array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]])
        >>> bbt.Bbox2D([(0, 1), (1, 3)], image_boundary=a.shape).assign(a, 1)
        >>> a
        array([[0., 1., 1.],
               [0., 0., 0.],
               [0., 0., 0.]])

        :param image: (np.ndarray or torch.Tensor) Source image ndarray should have shape (h, w) or (h, w, c)
                        Tensor should have shape (h, w) or (c, h, w) or (n, c, h, w)
        :param value: (int or float or np.ndarray or torch.Tensor) Value to fill the patch.
                        Int and float can assign to both np.ndarray and torch.Tensor. ndarray can only be assign to
                        ndarray, Tensor can only be assign to Tensor. Value Tensor must have less dimensions than image
                        Tensor. Acceptable shape:
                                image       |               value
                        -------------------------------------------------------
                        ndarry: (H, W)      |  int; float; ndarray (h, w)
                        ndarry: (H, W, c)   |  int; float; ndarray (h, w, c)
                        Tensor: (H, W)      |  int; float; Tensor (h, w)
                        Tensor: (c, H, W)   |  int; float; Tensor (h, w); Tensor (c, h, w)
                        Tensor: (n, c, H, W)|  int; float; Tensor (h, w); Tensor (c, h, w); Tensor (n, c, h, w)

        :param auto_fit: (bool) Whether automatically resize the value to be proper to fit into the target patch, only
                        take effects when value is ndarray or Tensor.
                        When disabled, (h, w) of Value must fit the shape of bbox.
                        When enabled, the value will be interpolated to spatial shape as this bbox. cv2.resize is used
                        for interpolation of ndarray (default interpolation method: cv2.INTER_AREA),
                        torch.nn.function.interpolate is used for interpolation of Tensor (default interpolation method:
                        'bilinear'). To change the interpolation method:
                        >>> bbt.resize_method_torch = 'nearest'
                        >>> bbt.resize_method = cv2.INTER_NEAREST

        :return: None
        """
        _check_image_size(image, self.boundary)

        if type(image) != type(value):
            if type(image) == np.ndarray and not (np.issubdtype(type(value), np.integer) or (np.issubdtype(type(value), np.floating))):
                raise Exception('Image type and value type are not matched, image: %s, value: %s' % (str(type(image)), str(type(value))))
            if enable_pytorch and type(image) == torch.Tensor and not (isinstance(type(value), int) or (isinstance(type(value), float))):
                raise Exception('Image type and value type are not matched, image: %s, value: %s' % (str(type(image)), str(type(value))))

        if type(image) == np.ndarray:
            type_ = 'numpy'
            image_boundary_ = image.shape[0:2]
        elif enable_pytorch and type(image) == torch.Tensor:
            type_ = 'torch'
            image_boundary_ = tuple(image.shape[-2::])
        else:
            raise Exception('Image must be either np.ndarray or torch.Tensor, got %s.' % str(type(image)))

        cropped_self = self.copy()
        if self.boundary is not None:
            if tuple(self.boundary) != image_boundary_:
                raise Exception('Bbox boundary %s does not fit image shape %s!' % (str(self.boundary), str(image_boundary_)))
        else:
            self.set_boundary(image_boundary_)

        if type(value) == int or type(value) == float:
            if type_ == 'numpy':
                if len(image.shape) == 2:
                    outshape = cropped_self.shape
                else:
                    outshape = cropped_self.shape + image.shape[2::]
                value = np.ones(outshape, dtype=image.dtype) * value
            else:
                outshape = tuple(image.shape[0:-2]) + cropped_self.shape
                value = torch.ones(outshape, dtype=image.dtype).to(image.device) * value

        elif type_ == 'numpy' and (not cropped_self.shape == value.shape[0:2]):
            if auto_fit:
                if enable_vc2:
                    value = cv2.resize(value, (cropped_self.shape[1], cropped_self.shape[0]), interpolation=resize_method)
                else:
                    raise Exception('Unable to import opencv for resize unpaired image to shape of bbox, box shape: '
                                    + str(cropped_self.shape) + ' image shape: ' + str(value.shape))
            else:
                raise Exception('Unpaired shape: box shape: ' + str(cropped_self.shape) + ' image shape: ' + str(value.shape))

        elif type_ == 'torch' and (not cropped_self.shape == value.shape[-2::]):
            if auto_fit:
                outshape = tuple(value.shape[0:-2]) + cropped_self.shape

                if len(value.shape) == 2:
                    value = value.unsqueeze(0)
                if len(value.shape) == 3:
                    value = value.unsqueeze(0)

                value = torch.nn.functional.interpolate(value, size=cropped_self.shape, mode=resize_method_torch)
                value = value.view(outshape)
            else:
                raise Exception('Unpaired shape: box shape: ' + str(cropped_self.shape) + ' image shape: ' + str(value.shape))

        # pair dimension of value and image. i,e. (5, 5) -> (1, 5, 5)
        if type_ == 'torch' and len(value.shape) < len(image.shape):
            for _ in range(len(image.shape) - len(value.shape)):
                value = value.unsqueeze(0)
        elif type_ == 'torch' and len(value.shape) > len(image.shape):
            raise Exception('Image should have more dimensions than value, Image has %d, Value has %d' % (len(image.shape), len(value.shape)))

        _assign_bbox(image, value, cropped_self.bbox, mode=type_)

    def remove_boundary(self):
        self.boundary = None
        return self

    def putback(self, dest, inbox):
        return _bbox_putback(dest, inbox, self.bbox)

    def pad(self, pad, axis=None, fix_size=False):
        if self.if_empty():
            return empty()

        out = self.copy()
        out.bbox = _box_pad(self.bbox, pad, self.boundary, axis=axis, fix_size=fix_size)
        return out

    def shift(self, value, axis=None, force=False):
        if self.if_empty():
            return empty()

        if not axis:
            if (type(value) == list or type(value) == tuple) and len(value) == 2:
                axis = [0, 1]
            else:
                raise Exception('Can only use default axis when shift value matched total axes!')
        out = self.copy()
        out.bbox = _box_shift(self.bbox, axis, value, self.boundary, force)
        return out

    def transpose(self):
        if self.if_empty():
            return empty()
        tem = Bbox2D(bbox=[self.bbox[1], self.bbox[0]], image_boundary=(self.boundary[1], self.boundary[0]))
        for att in self.attributes.keys():
            tem.__setattr__(att, self.__getattr__(att))
        return tem

    def _get_center(self):
        return (self.bbox[0][0] + self.bbox[0][1]) // 2, (self.bbox[1][0] + self.bbox[1][1]) // 2

    def set_shape(self, shape, center=None):
        if center is None:
            if self.if_empty():
                return self
            center = self.center
        new_box = [[int(center[0] - shape[0] // 2), int(center[0] + shape[0] - shape[0] // 2)],
                   [int(center[1] - shape[1] // 2), int(center[1] + shape[1] - shape[1] // 2)]]
        self.bbox = new_box

        if self.boundary:
            self._limit_to_boundary(boundary=self.boundary)

        return self

    def box_in_box(self, boxin):
        output = boxin.copy()
        output.boundary = None
        output = output.shift([-self.bbox[0][0], -self.bbox[1][0]], [0, 1], force=True)
        if not output:
            raise Exception('Out of boundary')
        output.set_boundary(self.shape)
        return output

    def box_out_box(self, boxin):
        output = boxin.copy()
        output.set_boundary(self.boundary)
        output = output.shift([self.bbox[0][0], self.bbox[1][0]], force=True)
        if not output:
            raise Exception('Out of boundary')
        return output

    def numpy(self, save_image_boundary=True, dtype=np.float32):
        if not self.boundary:
            save_image_boundary = False
        return list_box_to_numpy([self], save_image_boundary=save_image_boundary, dtype=dtype)[0]


def box_by_shape(shape, center, image_boundary=None):
    out = Bbox2D([(0, 1), (0, 1)], image_boundary=image_boundary)
    out.set_shape(shape, center)
    return out


def from_numpy(bbox, image_boundary=None, sorts=('y0', 'y1', 'x0', 'x1'), load_boundary_if_possible=True):
    """
    Create bbox from ndarray.
    Examples:
        >>> bbt.from_numpy(np.array([0, 5, 0, 4]))
        <class "Bbox2D", shape=[(0, 5), (0, 4)]>
        >>> bbt.from_numpy(np.array([[0, 0, 4, 5, 8, 9]]), sorts=('x0', 'y0', 'x1', 'y1'))
        [<class "Bbox2D", shape=[(0, 5), (0, 4)], boundary=[8, 9]>]

    :param bbox: (ndarray) array has shape (4, ) or (n, 4) without image boundary, or (6, ) or (n, 6) with image boundary.
    :param image_boundary: image boundary of bbox, it has higher priority than auto load boundary.
    :param sorts: the sort of coordinate in ndarray, could be 'y0', 'y1', 'x0', 'x1', 'h', 'w'; 'w' alongs 'x' axis.
                    default: ('y0', 'y1', 'x0', 'x1'). Notes: the default sort of PIL is ('x0', 'y0', 'x1', 'y1').
    :param load_boundary_if_possible: Automatically assign the boundary of the bbox if input ndarray has shape longer
                                      than 6.
    :return: Bbox2D if input ndarray is 1d array, list of Bbox2D if input ndarray is 2d array.
    """
    if type(bbox) == list:
        bbox = np.array(bbox)

    if len(bbox.shape) == 2:
        out_ = []
        for i in range(bbox.shape[0]):
            out_.append(from_numpy(bbox[i], image_boundary=image_boundary, sorts=sorts,
                                   load_boundary_if_possible=load_boundary_if_possible))
        return out_

    if 'w' in sorts or 'h' in sorts:
        bbox = bbox.copy()
        new_sorts = list(sorts).copy()
        if 'w' in sorts:
            if 'x0' in sorts:
                bbox[sorts.index('w')] = bbox[sorts.index('w')] + bbox[sorts.index('x0')]
                new_sorts[sorts.index('w')] = 'x1'
            elif 'x1' in sorts:
                bbox[sorts.index('w')] = bbox[sorts.index('x1')] - bbox[sorts.index('w')]
                new_sorts[sorts.index('w')] = 'x0'
        if 'h' in sorts:
            if 'y0' in sorts:
                bbox[sorts.index('h')] = bbox[sorts.index('h')] + bbox[sorts.index('y0')]
                new_sorts[sorts.index('h')] = 'y1'
            elif 'y1' in sorts:
                bbox[sorts.index('h')] = bbox[sorts.index('y1')] - bbox[sorts.index('h')]
                new_sorts[sorts.index('h')] = 'y0'
        sorts = new_sorts

    bbox = bbox.astype(np.int32)

    box_ = [(bbox[sorts.index('y0')], bbox[sorts.index('y1')]), (bbox[sorts.index('x0')], bbox[sorts.index('x1')])]
    if image_boundary is None and load_boundary_if_possible and bbox.size >= 6:
        image_boundary = [bbox[4], bbox[5]]
    return Bbox2D(box_, image_boundary=image_boundary)


def list_box_to_numpy(box_list, save_image_boundary=False, attributes=tuple(), dtype=np.float32):
    """
    Convert a list of Bbox2D to a 2d ndarray.
    :param box_list: (list) list of Bbox2d
    :param save_image_boundary: (bool) whether also include the boundary
    :param attributes: (tuple) attributes of bbox included in output
    :param dtype: (np dtype) the data type of output
    :return: 2-d ndarray
    """
    length = len(box_list)
    if save_image_boundary:
        width = 6 + len(attributes)
        output = np.zeros((length, width), dtype=dtype)
        for i in range(length):
            if box_list[i].if_empty():
                output[i, :] = -1
                continue
            if not box_list[i].boundary:
                raise Exception('Bbox %d have no boundary when save_image_boundary is enabled.' % i)
            output[i, 0:4] = np.array(box_list[i].bbox, dtype=dtype).ravel()
            output[i, 4:6] = np.array(box_list[i].boundary, dtype=dtype)

            for attr, j in zip(attributes, range(6, width)):
                output[i, j] = box_list[i].__getattr__(attr)
    else:
        width = 4 + len(attributes)
        output = np.zeros((length, width), dtype=dtype)
        for i in range(length):
            if box_list[i].if_empty():
                output[i, :] = -1
                continue
            output[i, 0:4] = np.array(box_list[i].bbox, dtype=dtype).ravel()
            for attr, j in zip(attributes, range(4, width)):
                output[i, j] = box_list[i].__getattr__(attr)
    return output


def pad(box, pad, axis=None, fix_size=False):
    return box.pad(pad, axis, fix_size)


def shift(box, axis, value, force=False):
    return box.shift(axis, value, force)


def empty():
    return Bbox2D(None, )


def full(boundary):
    if type(boundary) == np.ndarray and len(boundary.shape) > 1:
        boundary = (boundary.shape[0], boundary.shape[1])
    if enable_pytorch and type(boundary) == torch.Tensor and len(boundary.shape) > 1:
        boundary = (boundary.shape[-2], boundary.shape[-1])
    return Bbox2D(bbox=[(0, boundary[0]), (0, boundary[1])], image_boundary=boundary)


def nonzero(image):
    """
    Returns a bbox covers all non-zeros part of the image.
    :param image: for numpy: 2-D, 3-D ndarray
                  for torch: 2-D, 3-D, 4-D Tensor
    :return: Bbox2D with boundary
    """
    if type(image) == np.ndarray:
        if len(image.shape) == 3:
            image = np.max(np.abs(image), axis=2)
        non = np.nonzero(image)
        box = [(int(np.min(non[0])), int(np.max(non[0]) + 1)), (int(np.min(non[1])), int(np.max(non[1]) + 1))]
    elif enable_pytorch and type(image) == torch.Tensor:
        if len(image.shape) == 3:
            image = torch.max(torch.abs(image), dim=0)[0]
        if len(image.shape) == 4:
            image = torch.max(torch.max(torch.abs(image), dim=0)[0], dim=0)[0]
        non = torch.nonzero(image)
        box = [(int(torch.min(non[:, 0])), int(torch.max(non[:, 0]) + 1)), (int(torch.min(non[:, 1])), int(torch.max(non[:, 1]) + 1))]
    else:
        raise Exception('Unknown type of input image: %s' % type(image))
    return Bbox2D(bbox=box, image_boundary=image.shape)


def contain_points(points, image_boundary=None):
    if type(points) == list:
        points = np.array(points)

    if len(points.shape) == 1:
        points = np.reshape(points, (1, -1))

    return Bbox2D(list(zip(*[points.min(axis=0), points.max(axis=0)])), image_boundary=image_boundary)


def draw_bbox(image, box, boundary=None, fill=None, boundary_width=2, text=None):
    """
    Draw bbox on a image. The drawn boundary will halfly placed inside the bbox while halfly outside the bbox.
    IMPORTANT: input image will be changed, in order to keep original array unchange, please use image.copy()
    Notice current version only support ndarray.
    :param image: (ndarry) 2D image array, with size (W, H) or (W, H, C)
    :param box: (Bbox2D) box need to draw
    :param boundary: (list or tuple) boundary color, can be (R, G, B) or (R, G, B, A)
    :param fill: (list or tuple) fill color, can be (R, G, B) or (R, G, B, A)
    :param boundary_width: int
    :return: modified image array
    """
    dtype = image.dtype
    if not (boundary or fill):
        raise Exception('Must choose boundary or fill or both! Otherwise will return original image!')

    if box.if_empty():
        return image

    if len(image.shape) == 2:
        image = np.repeat(image.reshape(image.shape + (1,)), 3, axis=2)

    if fill:
        color = fill
        tem = box.apply(image)
        if len(color) == 4:
            tem[:, :, 0] = tem[:, :, 0] * (1 - color[3])
            tem[:, :, 0] = tem[:, :, 0] + color[3] * color[0]
            tem[:, :, 1] = tem[:, :, 1] * (1 - color[3])
            tem[:, :, 1] = tem[:, :, 1] + color[3] * color[1]
            tem[:, :, 2] = tem[:, :, 2] * (1 - color[3])
            tem[:, :, 2] = tem[:, :, 2] + color[3] * color[2]
        elif len(color) == 3:
            tem[:, :, 0] = color[0]
            tem[:, :, 1] = color[1]
            tem[:, :, 2] = color[2]
        else:
            raise Exception('Filling color must list or tuple with len 3 or 4, but get ' + str(color))

    if boundary:
        mask = np.zeros_like(image, dtype=np.uint8)
        box_outer = box.copy().pad(boundary_width // 2)
        box_inner = box.copy().pad(boundary_width // 2 - boundary_width)
        box_outer.assign(mask, 1)
        box_inner.assign(mask, 0)
        color = boundary
        if len(color) == 4:
            image = image - image * mask * color[3]
            mask[:, :, 0] = mask[:, :, 0] * color[0] * color[3]
            mask[:, :, 1] = mask[:, :, 1] * color[1] * color[3]
            mask[:, :, 2] = mask[:, :, 2] * color[2] * color[3]
            image += mask
        elif len(color) == 3:
            image -= image * mask
            mask[:, :, 0] *= color[0]
            mask[:, :, 1] *= color[1]
            mask[:, :, 2] *= color[2]
            image += mask

    if text:
        if not enable_PIL:
            raise Exception('To add text on bbox, PIL is required!')
        img = Image.fromarray(image)
        draw_one_annotation(img, (box.bbox[0][0], box.bbox[1][0]), text, )
        image[:] = np.array(img)

    return image.astype(dtype)


def draw_one_annotation(img, position, cate_s, font=ImageFont.load_default(), backgound_color='white'):
    y, x = position
    draw = ImageDraw.Draw(img)
    w, h = font.getsize(cate_s)
    draw.rectangle((x, y, x + w, y + h), fill=backgound_color)
    draw.text((x, y), cate_s, fill=(0, 0, 0), font=font)


def _box_shift(bbox, axis, value, boundary=None, force=False):
    if type(axis) == int and type(value) == int:
        axis = [axis]
        value = [value]

    bbox = [list(tem) for tem in bbox]
    for a, x in zip(axis, value):
        bbox[a][0] += x
        bbox[a][1] += x

        if boundary and bbox[a][0] < 0:
            if force:
                bbox[a][0] = 0
            else:
                bbox[a][1] -= bbox[a][0]
                bbox[a][0] -= bbox[a][0]

        if boundary and bbox[a][1] > boundary[a]:
            if force:
                bbox[a][1] = boundary[a]
            else:
                bbox[a][0] -= bbox[a][1] - boundary[a]
                bbox[a][1] -= bbox[a][1] - boundary[a]

    return bbox


def _bbox_putback(whole, inbox, bbox):
    if not (inbox.shape[0] == bbox[0][1] - bbox[0][0] and inbox.shape[1] == bbox[1][1] - bbox[1][0]):
        raise Exception(
            'Unpaired Shape, get box: %s, patch size: (%d, %d)' % (bbox.__str__(), inbox.shape[0], inbox.shape[1]))
    whole[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]] = inbox
    return whole


def _check_image_size(img, boundary):
    if boundary is None:
        return
    if img is None:
        raise Exception('Illegal input: None type image!')
    if enable_pytorch and type(img) == torch.Tensor:
        if len(img.shape) == 2:
            # (h, w)
            shape_ = img.shape
        elif len(img.shape) == 3:
            # (c, h, w)
            shape_ = img.shape[1::]
        elif len(img.shape) == 4:
            # (n, c, h, w)
            shape_ = img.shape[2::]
        else:
            raise Exception('Image could only be 2D (h, w), 3D (c, h, w), 4D (n, c, h, w) Tensor, but got shape' + str(img.shape))

    else:
        if len(img.shape) == 2:
            # (h, w)
            shape_ = img.shape
        elif len(img.shape) == 3:
            # (c, h, w)
            shape_ = img.shape[0:2]
        else:
            raise Exception(
                'Image could only be 2D (h, w), 3D (h, w, c) array, but got shape' + str(img.shape))

    if shape_[0] != boundary[0] or shape_[1] != boundary[1]:
        warnings.warn('Shape of input image %s does not fit boundary of bbox %s!' % (str(img.shape), str(boundary)))


def _apply_bbox_numpy(source, bbox):
    if len(source.shape) == 3:
        get = source[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], :]
    else:
        get = source[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]
    return get


def _apply_bbox_torch(source, bbox):
    if len(source.shape) == 4:
        get = source[:, :, bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]
    elif len(source.shape) == 3:
        get = source[:, bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]
    else:
        get = source[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]
    return get


def _apply_pad_numpy(tar_img, box_, **kwargs):
    out_shape = box_.shape

    if out_shape[0] // 2 - box_.center[0] > 0 or out_shape[1] // 2 - box_.center[1] > 0 or out_shape[0] // 2 + box_.center[
        0] - tar_img.shape[0] > 0 or out_shape[1] // 2 + box_.center[1] - tar_img.shape[1] > 0:
        if len(tar_img.shape) == 2:
            padding = (
            (max(out_shape[0] // 2 - box_.center[0], 0), max(out_shape[0] // 2 + box_.center[0] - tar_img.shape[0], 0)),
            (max(out_shape[1] // 2 - box_.center[1], 0), max(out_shape[1] // 2 + box_.center[1] - tar_img.shape[1], 0)))
        elif len(tar_img.shape) == 3:
            padding = (
            (max(out_shape[0] // 2 - box_.center[0], 0), max(out_shape[0] // 2 + box_.center[0] - tar_img.shape[0], 0)),
            (max(out_shape[1] // 2 - box_.center[1], 0), max(out_shape[1] // 2 + box_.center[1] - tar_img.shape[1], 0)),
            (0, 0))
        else:
            raise Exception(
            'Image could only be 2D (h, w), 3D (h, w, c) array, but got shape' + str(tar_img.shape))

        img_out = np.pad(tar_img, padding, **kwargs)
        box_ = box_.shift([padding[0][0], padding[1][0]])
        return _apply_bbox_numpy(img_out, box_)
    else:
        return _apply_bbox_numpy(tar_img, box_)


def _apply_pad_torch(tar_img, box_, **kwargs):
    out_shape = box_.shape

    if len(tar_img.shape) == 2:
        # (h, w)
        tar_shape = tar_img.shape
        tar_img = tar_img.unsqueeze(0).unsqueeze(0)
        in_dims = 2
    elif len(tar_img.shape) == 3:
        # (c, h, w)
        tar_shape = tar_img.shape[1::]
        tar_img = tar_img.unsqueeze(0)
        in_dims = 3
    elif len(tar_img.shape) == 4:
        # (n, c, h, w)
        tar_shape = tar_img.shape[2::]
        in_dims = 4
    else:
        raise Exception('Image could only be 2D (h, w), 3D (c, h, w), 4D (n, c, h, w) Tensor, but got shape' + str(tar_img.shape))

    if out_shape[0] // 2 - box_.center[0] > 0 or out_shape[1] // 2 - box_.center[1] > 0 or out_shape[0] // 2 + box_.center[
        0] - tar_shape[0] > 0 or out_shape[1] // 2 + box_.center[1] - tar_shape[1] > 0:
        # (padding_left, padding_right, padding_top, padding_bottom)
        padding = (
        max(out_shape[1] // 2 - box_.center[1], 0), max(out_shape[1] // 2 + box_.center[1] - tar_shape[1], 0),
        max(out_shape[0] // 2 - box_.center[0], 0), max(out_shape[0] // 2 + box_.center[0] - tar_shape[0], 0),)

        img_out = torch.nn.functional.pad(tar_img, padding, **kwargs)
        box_ = box_.shift([padding[2], padding[0]])

        get_img = _apply_bbox_torch(img_out, box_)

    else:
        get_img = _apply_bbox_torch(tar_img, box_)

    if in_dims == 2:
        return get_img.squeeze(0).squeeze(0)
    elif in_dims == 3:
        return get_img.squeeze(0)
    else:
        return get_img


def _assign_bbox(source, value, bbox, mode='numpy'):
    if mode == 'numpy':
        if len(source.shape) == 3:
            source[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], :] = value
        else:
            source[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]] = value
        return
    if mode == 'torch':
        if len(source.shape) == 4:
            source[:, :, bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]] = value
        elif len(source.shape) == 3:
            source[:, bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]] = value
        else:
            source[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]] = value
        return


def _box_pad(bbox, paddings, boundary=None, axis=None, fix_size=False):
    if axis is None:
        axis = np.arange(len(bbox)).tolist()

    if type(axis) == int:
        axis = [axis]

    if type(paddings) == int:
        paddings = [paddings] * len(bbox)

    bbox_out = [list(temp) for temp in bbox]
    if not boundary:
        for tem, padding in zip(axis, paddings):
            bbox_out[tem][0] -= padding
            bbox_out[tem][1] += padding

    elif not fix_size:
        for tem, padding in zip(axis, paddings):
            bbox_out[tem][0] = max(0, bbox_out[tem][0] - padding)
            bbox_out[tem][1] = min(boundary[tem], bbox_out[tem][1] + padding)

    else:
        for tem, padding in zip(axis, paddings):
            if bbox_out[tem][1] - bbox_out[tem][0] + 2 * padding >= boundary[tem]:
                raise Exception('Unable to apply expected size')
            length = bbox_out[tem][1] - bbox_out[tem][0] + 2 * padding
            bbox_out[tem][0] = max(0, bbox_out[tem][0] - padding)
            bbox_out[tem][1] = bbox_out[tem][0] + length
            if bbox_out[tem][1] > boundary[tem]:
                bbox_out[tem][1] = boundary[tem]
                bbox_out[tem][0] = bbox_out[tem][1] - length

    return bbox_out


def projection_function_by_boxes(source_box, target_box, compose=True, max_dim=2):
    # output: lambda x: (x - p0) * ratio + p1
    foos = []
    for axis_ in range(max_dim):
        p0 = source_box.bbox[axis_][0]
        p1 = target_box.bbox[axis_][0]
        ratio = target_box.shape[axis_] / source_box.shape[axis_]

        # hacking -> to avoid return same function
        foos.append(lambda x, p0=p0, ratio=ratio, p1=p1: (x - p0) * ratio + p1)

    if compose:
        return lambda mat_, dims=max_dim: torch.cat([foos[t](mat_[:, t:t + 1]) for t in range(dims)]).view(mat_.shape[::-1]).transpose(0, 1) \
            if enable_pytorch and type(mat_) == torch.Tensor else np.concatenate([foos[t](mat_[:, t:t + 1]) for t in range(dims)], axis=1)
    return foos


if __name__ == '__main__':
    a = box_by_shape((200, 200), (50, 50))
    im = np.ones((100, 100))
    print(a.apply(im).shape)
