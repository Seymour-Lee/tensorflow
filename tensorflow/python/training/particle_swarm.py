# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ParticleSwarm for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from tensorflow.python.framework import ops
# from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import resource_variable_ops
# from tensorflow.python.training import optimizer
# from tensorflow.python.training import training_ops

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import random_ops
from tensorflow.python.training import optimizer

from tensorflow.python.client import session as tf_session



class ParticleSwarmOptimizer(optimizer.Optimizer):
  """Optimizer that implements the AMSGrad algorithm.
    See (https://openreview.net/pdf?id=ryQu7f-RZ)
    
  """

  # Values for gate_gradients.
  GATE_NONE = 0
  GATE_OP = 1
  GATE_GRAPH = 2

  def __init__(self, learning_rate=0.01, m=20, w=0.8, c1=2, c2=2, use_locking=False, name="ParticleSwarm"):
    """Construct a new AMSGrad optimizer.
    Args:
        learning_rate: A Tensor or a floating point value. The 
            learning rate.
        beta1: A float value or a constant float tensor. The 
            exponential decay rate for the 1st moment estimates.
        beta2: A float value or a constant float tensor. The
            exponential decay rate for the 2nd moment estimates.
        epsilon: A small constant for numerical stability. 
        name: Optional name for the operations created when applying gradients.
        Defaults to "AMSGrad".
    """
    super(ParticleSwarmOptimizer, self).__init__(use_locking, name)
    self._lr = learning_rate

    self._lr_t = None

    self._m = m
    self._w = w
    self._c1 = c1
    self._c2 = c2

    self._m_t = None
    self._w_t = None
    self._c1_t = None
    self._c2_t = None

    self._loss = None
    self._var_list = None

    self._calculate_list = None

  def _create_slots(self, var_list):
    # TO DO: for each var, create following slots: partile[m], pbest, gbest DONE!

    # Create slots for the first and second moments.
    for v in var_list :
        self._zeros_slot(v, "gBest", self._name)
        self._zeros_slot(v, "pBest", self._name)
        self._zeros_slot(v, "r1", self._name)
        self._zeros_slot(v, "r2", self._name)

        for i in range(self._m):
          self._zeros_slot(v, "p_" + str(i), self._name)
          self._zeros_slot(v, "v_" + str(i), self._name)

  def _prepare(self):
    # TO DO: instantiaze all _var_name_t in __init__() DONE!
    self._m_t = ops.convert_to_tensor(self._m, name="m")
    self._w_t = ops.convert_to_tensor(self._w, name="w")
    self._c1_t = ops.convert_to_tensor(self._c1, name="c1")
    self._c2_t = ops.convert_to_tensor(self._c2, name="c2")

  def _apply_dense(self, grad, var):
    # print("var is :")
    # print(var)

    # TO DO: cast all var_name_t into math_ops DONE!
    m_t = math_ops.cast(self._m_t, var.dtype.base_dtype)
    w_t = math_ops.cast(self._w_t, var.dtype.base_dtype)
    c1_t = math_ops.cast(self._c1_t, var.dtype.base_dtype)
    c2_t = math_ops.cast(self._c2_t, var.dtype.base_dtype)

    # print("in _apply_dense():")
    # print(var_update, m_t, v_t, vhat_t)
    # print("-----------------------")
    # print(control_flow_ops.group(*[var_update, m_t, v_t, vhat_t]))
    # print("end _apply_dense()")

    # TO DO: calculate current var-d DONE!
    calculate_list = []

    gBest = self.get_slot(var, "gBest")
    var_shape = gBest.get_shape()
    # print(gBest.get_shape())
    # print(var_shape)
    pBest = self.get_slot(var, "pBest")
    r1 = self.get_slot(var, "r1")
    r2 = self.get_slot(var, "r2")

    # calculate_list might be insert in zero position, rather than append to the list tail

    for d in range(self._m):
      v_d = self.get_slot(var, "v_" + str(d))
      p_d = self.get_slot(var, "p_" + str(d))
      # The following two lines may have problem, because when running this positon, the compute graph is not running
      # The following two lines have been changed, but I am not sure they are correct
      r1_t = state_ops.assign(r1, random_ops.random_uniform(var_shape, 0, 1)) # find out how to generate a random value, which belongs to (0, 1)
      r2_t = state_ops.assign(r2, random_ops.random_uniform(var_shape, 0, 1))
      # vi,d = w * vi,d + c1 * r1 * (pBest - var) + c2 * r2 * (gBest - var)
      v_d_t = state_ops.assign(v_d, w_t * v_d + c1_t * r1_t * (pBest - var) + c2_t * r2_t * (gBest - var))
      # var = var + vi,d
      p_d_t = state_ops.assign_add(p_d, v_d_t)
      calculate_list.append(p_d_t)
      calculate_list.append(v_d_t)
      # print(p_d_t, v_d_t)
      # print(calculate_list)

    return control_flow_ops.group(*calculate_list)

  def _resource_apply_dense(self, grad, handle):
    return training_ops.resource_apply_particle_swarm(
        handle.handle, math_ops.cast(self._learning_rate_tensor,
                                     grad.dtype.base_dtype),
        grad, use_locking=self._use_locking)

  def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
    return resource_variable_ops.resource_scatter_add(
        handle.handle, indices, -grad * self._learning_rate)

  def _apply_sparse_duplicate_indices(self, grad, var):
    delta = ops.IndexedSlices(
        grad.values *
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad.indices, grad.dense_shape)
    return var.scatter_sub(delta, use_locking=self._use_locking)

  def minimize(self, loss, global_step=None, var_list=None,
               gate_gradients=GATE_OP, aggregation_method=None,
               colocate_gradients_with_ops=False, name=None,
               grad_loss=None):
    self._loss = loss
    print("in child class : minimizer()")
    print(id(self._loss), id(loss))


    if self._var_list is None:
      self._var_list = (
          variables.trainable_variables() +
          ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
    else:
      self._var_list = nest.flatten(self._var_list)
    # pylint: disable=protected-access
    self._var_list += ops.get_collection(ops.GraphKeys._STREAMING_MODEL_PORTS)
    print(self._var_list)

    ret = super(ParticleSwarmOptimizer, self).minimize(loss, global_step, var_list,
               gate_gradients, aggregation_method,
               colocate_gradients_with_ops, name,
               grad_loss)
    print(ret)
    return ret
    

  def _finish(self, update_ops, name_scope):
    """Do what is needed to finish the update.

    This is called with the `name_scope` using the "name" that
    users have chosen for the application of gradients.

    Args:
      update_ops: List of `Operation` objects to update variables.  This list
        contains the values returned by the `_apply_dense()` and
        `_apply_sparse()` calls.
      name_scope: String.  Name to use for the returned operation.

    Returns:
      The operation to apply updates.
    """
    print("in child class : _finish()")
    # print(update_ops)
    # print(*update_ops)
    sess = tf_session.Session()
    sess.run(variables.global_variables_initializer())

    calculate_list = []

    # calculate loss(gBest)
    for var in self._var_list:
        gBest_d = self.get_slot(var, "gBest")
        gBest_t = state_ops.assign(var, gBest_d)
        # calculate_list.insert(0, gBest_t)
        calculate_list.append(gBest_t)
    loss_gBest = sess.run(self._loss)

    for i in range(self._m):
      # calculate loss(p_i) for particle i
      for var in self._var_list:
        p_i_d = self.get_slot(var, "p_" + str(i))
        var_t = state_ops.assign(var, p_i_d)
        # calculate_list.insert(0, var_t)
        calculate_list.append(var_t)
      loss_p_i = sess.run(self._loss)
      print("loss_p_i", loss_p_i)

      # calculate loss(pBest_i) for particle i
      for var in self._var_list:
        pBest_i_d = self.get_slot(var, "pBest")
        pBest_t = state_ops.assign(var, pBest_i_d)
        # calculate_list.insert(0, pBest_t)
        calculate_list.append(pBest_t)
      loss_pBest_i = sess.run(self._loss)
      print("loss_pBest_i", loss_pBest_i)

      # self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")
      # math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

      for var in self._var_list:
        pBest_i_d = self.get_slot(var, "pBest")
        p_i_d = self.get_slot(var, "p_" + str(i))
        gBest_d = self.get_slot(var, "gBest")
        # cast loss_value into tensor
        pBest_t = state_ops.assign(pBest_i_d, math_ops.cast(ops.convert_to_tensor(loss_p_i < loss_pBest_i, name="p_pBest_0"), var.dtype.base_dtype) * p_i_d)
        # calculate_list.insert(0, pBest_t)
        calculate_list.append(pBest_t)
        loss_pBest_i = math_ops.minimum(loss_p_i, loss_pBest_i)
        gBest_t = state_ops.assign(gBest_d, math_ops.cast(ops.convert_to_tensor(loss_p_i < loss_pBest_i, name="p_pBest_1"), var.dtype.base_dtype) 
                                            * math_ops.cast(ops.convert_to_tensor(loss_pBest_i < loss_gBest, name="pBest_gBest"), var.dtype.base_dtype) 
                                            * p_i_d)
        # calculate_list.insert(0, gBest_t)
        calculate_list.append(gBest_t)
        loss_gBest = math_ops.minimum(loss_pBest_i, loss_gBest)
      print("loss_gBest", loss_gBest)

    calculate_list = control_flow_ops.group(*calculate_list)
    print("------------------")
    # print(calculate_list)
    print("------------------")
    # update_ops.insert(0, calculate_list)
    update_ops.append(calculate_list)
    # print(*update_ops)    
    return control_flow_ops.group(*update_ops, name=name_scope)

  def _random_slot(self, var, slot_name, op_name):
    """Find or create a slot initialized with random value.

    Args:
      var: A `Variable` object.
      slot_name: Name for the slot.
      op_name: Name to use when scoping the Variable that
        needs to be created for the slot.

    Returns:
      A `Variable` object.
    """
    # TO DO: re-write the following code to give slots a random value
    named_slots = self._slot_dict(slot_name)
    if _var_key(var) not in named_slots:
      named_slots[_var_key(var)] = slot_creator.create_zeros_slot(var, op_name)
    return named_slots[_var_key(var)]
