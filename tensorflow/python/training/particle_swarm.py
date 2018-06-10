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
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import adam
from tensorflow.python.training import slot_creator

from tensorflow.python.client import session as tf_session

def _var_key(var):
  if context.in_eager_mode():
    return var._shared_name  # pylint: disable=protected-access
  return (var.op.graph, var.op.name)

class ParticleSwarmOptimizer(optimizer.Optimizer):
  """
    Optimizer that implements the Particle Swarm algorithm.
  """

  # Values for gate_gradients.
  GATE_NONE = 0
  GATE_OP = 1
  GATE_GRAPH = 2

  def __init__(self, learning_rate=0.01, m=3, w=1e-4, c1=1e-4, c2=1e-4, use_locking=False, name="ParticleSwarm"):
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
    self._pBest_loss = None
    self._gBest_loss = None
    self._pc_loss = None

    self._calculate_list = None

  def _create_slots(self, var_list):
    # TO DO: for each var, create following slots: partile[m], pbest, gbest DONE!
    self._pBest_loss = variables.Variable(array_ops.zeros([self._m]))
    self._gBest_loss = variables.Variable(array_ops.zeros([1]))
    self._pc_loss = variables.Variable(array_ops.zeros([self._m]))

    # Create slots for the first and second moments.
    for v in var_list :
        self._zeros_slot(v, "gBest", self._name)
        self._random_slot(v, "r1", self._name)
        self._random_slot(v, "r2", self._name)

        for i in range(self._m):
          self._zeros_slot(v, "pBest_" + str(i), self._name)
          self._zeros_slot(v, "p_" + str(i), self._name)
          self._random_slot(v, "v_" + str(i), self._name)

  def _prepare(self):
    # TO DO: instantiaze all _var_name_t in __init__() DONE!
    self._m_t = ops.convert_to_tensor(self._m, name="m")
    self._w_t = ops.convert_to_tensor(self._w, name="w")
    self._c1_t = ops.convert_to_tensor(self._c1, name="c1")
    self._c2_t = ops.convert_to_tensor(self._c2, name="c2")

  def _apply_dense(self, grad, var):
    # TO DO: cast all var_name_t into math_ops DONE!
    m_t = math_ops.cast(self._m_t, var.dtype.base_dtype)
    w_t = math_ops.cast(self._w_t, var.dtype.base_dtype)
    c1_t = math_ops.cast(self._c1_t, var.dtype.base_dtype)
    c2_t = math_ops.cast(self._c2_t, var.dtype.base_dtype)

    # TO DO: calculate current var-d DONE!
    calculate_list = []

    gBest = self.get_slot(var, "gBest")
    var_shape = gBest.get_shape()
    r1 = self.get_slot(var, "r1")
    r2 = self.get_slot(var, "r2")

    # calculate_list might be insert in zero position, rather than append to the list tail
    for d in range(self._m):
      v_d = self.get_slot(var, "v_" + str(d))
      p_d = self.get_slot(var, "p_" + str(d))
      pBest_d = self.get_slot(var, "pBest_" + str(d))
      
      r1_t = state_ops.assign(r1, random_ops.random_uniform(var_shape, 0, 2)) # how to determine the scope of random value
      r2_t = state_ops.assign(r2, random_ops.random_uniform(var_shape, 0, 2))
      # vi,d = w * vi,d + c1 * r1 * (pBest - var) + c2 * r2 * (gBest - var)
      v_d_t = state_ops.assign(v_d, w_t * v_d + c1_t * r1_t * (pBest_d - var) + c2_t * r2_t * (gBest - var))
      # var = var + vi,d
      p_d_t = state_ops.assign_add(p_d, v_d_t)
      # calculate_list.append(p_d_t)
      # calculate_list.append(v_d_t)
      calculate_list.insert(0, r1_t)
      calculate_list.insert(0, r2_t)
      calculate_list.insert(0, p_d_t)
      calculate_list.insert(0, v_d_t)

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
    N_PARTICLES = 32
    P_BEST_FACTOR = 0.6
    G_BEST_FACTOR = 0.8
    L_BEST_FACTOR = 0.7
    # Velocity Decay specifies the multiplier for the velocity update
    VELOCITY_DECAY = 1
    # Velocity Restrict is computationally slightly more expensive
    VELOCITY_RESTRICT = True
    MAX_VEL = 0.005
    # Allows to decay the maximum velocity with each update
    # Useful if the network needs very fine tuning towards the end
    MAX_VEL_DECAY = 1

    # Hybrid Parameters
    HYBRID = True
    LEARNING_RATE = 0.1
    LBPSO = False


    # Other Params
    HIDDEN_LAYERS = args.hl

    CHI = 1  # Temporary Fix


    # Basic Neural Network Definition
    # Simple feedforward Network
    LAYERS = [N_IN] + HIDDEN_LAYERS # + [1]
    print('Network Structure\t:', LAYERS)


    t_VELOCITY_DECAY = tf.constant(value=VELOCITY_DECAY,
                                   dtype=tf.float32,
                                   name='vel_decay')
    t_MVEL = tf.Variable(MAX_VEL,
                         dtype=tf.float32,
                         name='vel_restrict',
                         trainable=False)
    
    # MULTI-PARTICLE NEURAL NETS

    losses = []
    nets = []
    pweights = []
    pbiases = []
    vweights = []
    vbiases = []

    random_values = []

    # Positional Updates
    bias_updates = []
    weight_updates = []

    # Velocity Updates
    vweight_updates = []
    vbias_updates = []

    # Fitness Updates
    fit_updates = []


    # Control Updates - Controling PSO inside tf.Graph
    control_updates = []

    # Hybrid Updates - Using of PSO + Traditional Approaches
    hybrid_updates = []

    gweights = None
    gbiases = None
    gfit = None

    if not LBPSO:
        gweights = []
        gbiases = []
        gfit = tf.Variable(float("inf"), name='gbestfit', trainable=False)

    # TODO:Parellized the following loop
    # TODO:See if the Conditional Function Lambdas can be optimized

    for pno in range(N_PARTICLES):
        weights = []
        biases = []
        pweights = []
        pbiases = []
        lweights = None
        lbiases = None
        if LBPSO:
            # Initialize the list
            lweights = []
            lbiases = []
        pbestrand = tf.Variable(tf.random_uniform(
            shape=[], maxval=P_BEST_FACTOR),
            name='pno' + str(pno + 1) + 'pbestrand',
            trainable=False)
        gbestrand = None
        lbestrand = None
        if not LBPSO:
            gbestrand = tf.Variable(tf.random_uniform(
                shape=[], maxval=G_BEST_FACTOR),
                name='pno' + str(pno + 1) + 'gbestrand',
                trainable=False)
        else:
            lbestrand = tf.Variable(tf.random_uniform(
                shape=[], maxval=L_BEST_FACTOR),
                name='pno' + str(pno + 1) + 'lbestrand',
                trainable=False)

        # Append the random values so that the initializer can be called again
        random_values.append(pbestrand)
        if not LBPSO:
            random_values.append(gbestrand)
        else:
            random_values.append(lbestrand)
        pfit = None
        with tf.variable_scope("fitnessvals", reuse=tf.AUTO_REUSE):
            init = tf.constant(float("inf"))
            pfit = tf.get_variable(name=str(pno + 1),
                                   initializer=init)

        pfit = tf.Variable(float("inf"), name='pno' + str(pno + 1) + 'fit')

        localfit = None
        if LBPSO:
            localfit = tf.Variable(float("inf"), name='pno' + str(pno + 1) + 'lfit')
        net = net_in
        # Define the parameters

        for idx, num_neuron in enumerate(LAYERS[1:]):
            layer_scope = 'pno' + str(pno + 1) + 'fc' + str(idx + 1)
            net, pso_tupple = fc(input_tensor=net,
                                 n_output_units=num_neuron,
                                 activation_fn='softmax', # 'sigmoid',
                                 scope=layer_scope,
                                 uniform=True)
            w, b, pw, pb, vw, vb = pso_tupple
            vweights.append(vw)
            vbiases.append(vb)
            weights.append(w)
            biases.append(b)
            pweights.append(pw)
            pbiases.append(pb)
            lw = None
            lb = None
            if LBPSO:
                lw = tf.Variable(pw.initialized_value(), name='lbest_w')
                lb = tf.Variable(pb.initialized_value(), name='lbest_b')
                lbiases.append(lb)
                lweights.append(lw)

            # Multiply by the Velocity Decay
            nextvw = tf.multiply(vw, t_VELOCITY_DECAY)
            nextvb = tf.multiply(vb, t_VELOCITY_DECAY)

            # Differences between Particle Best & Current
            pdiffw = tf.multiply(tf.subtract(pw, w), pbestrand)
            pdiffb = tf.multiply(tf.subtract(pb, b), pbestrand)

            # Differences between the Local Best & Current
            ldiffw = None
            ldiffb = None
            if LBPSO:
                ldiffw = tf.multiply(tf.subtract(lw, w), lbestrand)
                ldiffb = tf.multiply(tf.subtract(lb, w), lbestrand)

            # Define & Reuse the GBest
            gw = None
            gb = None
            if not LBPSO:
                with tf.variable_scope("gbest", reuse=tf.AUTO_REUSE):
                    gw = tf.get_variable(name='fc' + str(idx + 1) + 'w',
                                         shape=[LAYERS[idx], LAYERS[idx + 1]],
                                         initializer=tf.zeros_initializer)

                    gb = tf.get_variable(name='fc' + str(idx + 1) + 'b',
                                         shape=[LAYERS[idx + 1]],
                                         initializer=tf.zeros_initializer)

            # If first Particle add to Global Else it is already present
            if pno == 0 and not LBPSO:
                gweights.append(gw)
                gbiases.append(gb)
            gdiffw = None
            gdiffb = None
            # Differences between Global Best & Current
            if not LBPSO:
                gdiffw = tf.multiply(tf.subtract(gw, w), gbestrand)
                gdiffb = tf.multiply(tf.subtract(gb, b), gbestrand)
            else:
                ldiffw = tf.multiply(tf.subtract(lw, w), lbestrand)
                ldiffb = tf.multiply(tf.subtract(lb, b), lbestrand)

            vweightdiffsum = None
            vbiasdiffsum = None
            if LBPSO:
                vweightdiffsum = tf.multiply(
                    tf.add_n([nextvw, pdiffw, ldiffw]),
                    CHI)
                vbiasdiffsum = tf.multiply(tf.add_n([nextvb, pdiffb, ldiffb]), CHI)
            else:
                vweightdiffsum = tf.add_n([nextvw, pdiffw, gdiffw])
                vbiasdiffsum = tf.add_n([nextvb, pdiffb, gdiffb])

            vweight_update = None
            if VELOCITY_RESTRICT is False:
                vweight_update = tf.assign(vw, vweightdiffsum, validate_shape=True)
            else:
                vweight_update = tf.assign(vw, maxclip(vweightdiffsum, t_MVEL),
                                           validate_shape=True)

            vweight_updates.append(vweight_update)
            vbias_update = None
            if VELOCITY_RESTRICT is False:
                vbias_update = tf.assign(vb, vbiasdiffsum, validate_shape=True)
            else:
                vbias_update = tf.assign(vb, maxclip(vbiasdiffsum, t_MVEL),
                                         validate_shape=True)

            vbias_updates.append(vbias_update)
            weight_update = tf.assign(w, w + vw, validate_shape=True)
            weight_updates.append(weight_update)
            bias_update = tf.assign(b, b + vb, validate_shape=True)
            bias_updates.append(bias_update)

        # Define loss for each of the particle nets
        loss = tf.nn.l2_loss(net - label)
        # loss = tf.reduce_mean(-tf.reduce_sum(label*tf.log(net), reduction_indices=[1]))
        particlebest = tf.cond(loss < pfit, lambda: loss, lambda: pfit)
        fit_update = tf.assign(pfit, particlebest, validate_shape=True)
        fit_updates.append(fit_update)
        if not LBPSO:
            globalbest = tf.cond(loss < gfit, lambda: loss, lambda: gfit)
            fit_update = tf.assign(gfit, globalbest, validate_shape=True)
            fit_updates.append(fit_update)
        control_update = tf.assign(t_MVEL, tf.multiply(t_MVEL, MAX_VEL_DECAY),
                                   validate_shape=True)
        control_updates.append(control_update)
        if HYBRID:
            optimizer = adam.AdamOptimizer(learning_rate=LEARNING_RATE)
            hybrid_update = optimizer.minimize(loss)
            hybrid_updates.append(hybrid_update)

        # Multiple Length Checks
        assert len(weights) == len(biases)
        assert len(pweights) == len(pbiases)
        assert len(pweights) == len(weights)

        for i in range(len(weights)):
            # Particle Best
            pweight = tf.cond(loss <= pfit, lambda: weights[
                              i], lambda: pweights[i])
            fit_update = tf.assign(pweights[i], pweight, validate_shape=True)
            fit_updates.append(fit_update)
            pbias = tf.cond(loss <= pfit, lambda: biases[i], lambda: pbiases[i])
            fit_update = tf.assign(pbiases[i], pbias, validate_shape=True)
            fit_updates.append(fit_update)

            if LBPSO:
                lneigh = (pno - 1) % N_PARTICLES
                rneigh = (pno + 1) % N_PARTICLES
                lneighscope = 'pno' + str(lneigh + 1) + 'fc' + str(i + 1)
                rneighscope = 'pno' + str(rneigh + 1) + 'fc' + str(i + 1)
                lneigh_weight = None
                lneigh_bias = None
                rneigh_weight = None
                rneigh_bias = None
                lfit = None
                rfit = None

                with tf.variable_scope(lneighscope, reuse=tf.AUTO_REUSE):
                    lneigh_weight = tf.get_variable(
                        shape=[LAYERS[i], LAYERS[i + 1]],
                        name='pbest_w',
                        initializer=tf.random_uniform_initializer)
                    # [LAYERS[idx + 1]]
                    lneigh_bias = tf.get_variable(
                        shape=[LAYERS[i + 1]],
                        name='pbest_b',
                        initializer=tf.random_uniform_initializer)
                with tf.variable_scope(rneighscope, reuse=tf.AUTO_REUSE):
                    rneigh_weight = tf.get_variable(
                        shape=[LAYERS[i], LAYERS[i + 1]],
                        name='pbest_w',
                        initializer=tf.random_uniform_initializer)
                    # [LAYERS[idx + 1]]
                    rneigh_bias = tf.get_variable(
                        shape=[LAYERS[i + 1]],
                        name='pbest_b',
                        initializer=tf.random_uniform_initializer)

                with tf.variable_scope("fitnessvals", reuse=tf.AUTO_REUSE):
                    init = tf.constant(float("inf"))
                    lfit = tf.get_variable(name=str(lneigh + 1), initializer=init)
                    rfit = tf.get_variable(name=str(rneigh + 1), initializer=init)

                new_local_weight = None
                new_local_bias = None
                new_local_fit = None

                # Deal with Local Fitness
                neighbor_best_fit = tf.cond(lfit <= rfit,
                                            lambda: lfit, lambda: rfit)
                particle_best_fit = tf.cond(pfit <= localfit,
                                            lambda: pfit, lambda: localfit)
                best_fit = tf.cond(neighbor_best_fit <= particle_best_fit,
                                   lambda: neighbor_best_fit,
                                   lambda: particle_best_fit)
                fit_update = tf.assign(localfit, best_fit, validate_shape=True)
                fit_updates.append(fit_update)

                # Deal with Local Best Weights
                neighbor_best_weight = tf.cond(lfit <= rfit,
                                               lambda: lneigh_weight,
                                               lambda: rneigh_weight)
                particle_best_weight = tf.cond(pfit <= localfit,
                                               lambda: pweights[i],
                                               lambda: lweights[i])
                best_weight = tf.cond(neighbor_best_fit <= particle_best_fit,
                                      lambda: neighbor_best_weight,
                                      lambda: particle_best_weight)
                fit_update = tf.assign(
                    lweights[i], best_weight, validate_shape=True)
                fit_updates.append(fit_update)

                # Deal with Local Best Biases
                neighbor_best_bias = tf.cond(lfit <= rfit,
                                             lambda: lneigh_bias,
                                             lambda: rneigh_bias)
                particle_best_bias = tf.cond(pfit <= localfit,
                                             lambda: pbiases[i],
                                             lambda: lbiases[i])
                best_bias = tf.cond(neighbor_best_fit <= particle_best_fit,
                                    lambda: neighbor_best_bias,
                                    lambda: particle_best_bias)
                fit_update = tf.assign(lbiases[i], best_bias, validate_shape=True)
                fit_updates.append(fit_update)

            if not LBPSO:
                # Global Best
                gweight = tf.cond(loss <= gfit,
                                  lambda: weights[i],
                                  lambda: gweights[i])
                fit_update = tf.assign(gweights[i], gweight, validate_shape=True)
                fit_updates.append(fit_update)
                gbias = tf.cond(loss <= gfit,
                                lambda: biases[i],
                                lambda: gbiases[i])
                fit_update = tf.assign(gbiases[i], gbias, validate_shape=True)
                fit_updates.append(fit_update)

        # Update the lists
        nets.append(net)
        losses.append(loss)
        print_prog_bar(iteration=pno + 1,
                       total=N_PARTICLES,
                       suffix=str_memusage('M'))
    print(nets)

    msgtime('Completed\t\t:')

    # Initialize the entire graph
    init = tf.global_variables_initializer()
    msgtime('Graph Init Successful\t:')

    # Define the updates which are to be done before each iterations
    random_updates = [r.initializer for r in random_values]
    updates = weight_updates + bias_updates + \
        random_updates + vbias_updates + vweight_updates + \
        fit_updates + control_updates + hybrid_updates
    req_list = None
    if not LBPSO:
        req_list = losses, updates, gfit, gbiases, vweights, vbiases, gweights
    else:
        req_list = losses, updates, vweights, vbiases
    pass
    

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
    # TO DO: re-write the following code to give slots a random value DONE!
    named_slots = self._slot_dict(slot_name)
    if _var_key(var) not in named_slots:
      # named_slots[_var_key(var)] = slot_creator.create_zeros_slot(var, op_name)
      var_shape = var.get_shape()
      val_random = random_ops.random_uniform(var_shape, 0, 2)
      named_slots[_var_key(var)] = slot_creator.create_slot(var, val_random, op_name)
    return named_slots[_var_key(var)]
