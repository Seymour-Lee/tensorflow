import tensorflow as tf
from tensorflow.contrib.pso.python.utils import msgtime, str_memusage, print_prog_bar, fcn_stats, chical, maxclip, fc

class ParticleSwarmOptimizer(object):
    def __init__(self,
                 xorn,
                 bs,
                 pno,
                 pbest,
                 gbest,
                 lbest,
                 veldec,
                 vr,
                 mv,
                 mvdec,
                 hybrid,
                 lr,
                 lbpso,
                 iters,
                 hl,
                 pi,
                 ):
        
        # XOR Dataset Params
        self.N_IN = xorn
        self.N_BATCHSIZE = bs


        # PSO params
        self.N_PARTICLES = pno
        self.P_BEST_FACTOR = pbest
        self.G_BEST_FACTOR = gbest
        self.L_BEST_FACTOR = lbest
        # Velocity Decay specifies the multiplier for the velocity update
        self.VELOCITY_DECAY = veldec
        # Velocity Restrict is computationally slightly more expensive
        self.VELOCITY_RESTRICT = vr
        self.MAX_VEL = mv
        # Allows to decay the maximum velocity with each update
        # Useful if the network needs very fine tuning towards the end
        self.MAX_VEL_DECAY = mvdec

        # Hybrid Parameters
        self.HYBRID = hybrid
        self.LEARNING_RATE = lr
        self.LBPSO = lbpso


        # Other Params
        self.N_ITERATIONS = iters
        self.HIDDEN_LAYERS = hl
        self.PRINT_ITER = pi

        # Chi cannot be used for low value of pbest & lbest factors
        # self.CHI = chical(self.P_BEST_FACTOR, self.L_BEST_FACTOR)
        self.CHI = 1  # Temporary Fix


        # Basic Neural Network Definition
        # Simple feedforward Network
        self.LAYERS = [self.N_IN] + self.HIDDEN_LAYERS # + [1]
        print('Network Structure\t:', self.LAYERS)


        self.t_VELOCITY_DECAY = tf.constant(value=self.VELOCITY_DECAY,
                                    dtype=tf.float32,
                                    name='vel_decay')
        self.t_MVEL = tf.Variable(self.MAX_VEL,
                            dtype=tf.float32,
                            name='vel_restrict',
                            trainable=False)

        self.losses = []
        self.nets = []
        self.pweights = []
        self.pbiases = []
        self.vweights = []
        self.vbiases = []

        self.random_values = []

        # Positional Updates
        self.bias_updates = []
        self.weight_updates = []

        # Velocity Updates
        self.vweight_updates = []
        self.vbias_updates = []

        # Fitness Updates
        self.fit_updates = []


        # Control Updates - Controling PSO inside tf.Graph
        self.control_updates = []

        # Hybrid Updates - Using of PSO + Traditional Approaches
        self.hybrid_updates = []

        self.gweights = None
        self.gbiases = None
        self.gfit = None

        if not self.LBPSO:
            self.gweights = []
            self.gbiases = []
            self.gfit = tf.Variable(float("inf"), name='gbestfit', trainable=False)

    def minimize(self, net_in, label):
        for pno in range(self.N_PARTICLES):
            self.weights = []
            self.biases = []
            self.pweights = []
            self.pbiases = []
            self.lweights = None
            self.lbiases = None
            if self.LBPSO:
                # Initialize the list
                self.lweights = []
                self.lbiases = []
            pbestrand = tf.Variable(tf.random_uniform(
                shape=[], maxval=self.P_BEST_FACTOR),
                name='pno' + str(pno + 1) + 'pbestrand',
                trainable=False)
            gbestrand = None
            lbestrand = None
            if not self.LBPSO:
                gbestrand = tf.Variable(tf.random_uniform(
                    shape=[], maxval=self.G_BEST_FACTOR),
                    name='pno' + str(pno + 1) + 'gbestrand',
                    trainable=False)
            else:
                lbestrand = tf.Variable(tf.random_uniform(
                    shape=[], maxval=self.L_BEST_FACTOR),
                    name='pno' + str(pno + 1) + 'lbestrand',
                    trainable=False)

            # Append the random values so that the initializer can be called again
            self.random_values.append(pbestrand)
            if not self.LBPSO:
                self.random_values.append(gbestrand)
            else:
                self.random_values.append(lbestrand)
            pfit = None
            with tf.variable_scope("fitnessvals", reuse=tf.AUTO_REUSE):
                init = tf.constant(float("inf"))
                pfit = tf.get_variable(name=str(pno + 1),
                                       initializer=init)

            pfit = tf.Variable(float("inf"), name='pno' + str(pno + 1) + 'fit')

            localfit = None
            if self.LBPSO:
                localfit = tf.Variable(float("inf"), name='pno' + str(pno + 1) + 'lfit')
            net = net_in
            # Define the parameters

            for idx, num_neuron in enumerate(self.LAYERS[1:]):
                layer_scope = 'pno' + str(pno + 1) + 'fc' + str(idx + 1)
                net, pso_tupple = fc(input_tensor=net,
                                     n_output_units=num_neuron,
                                     activation_fn='softmax', # 'sigmoid',
                                     scope=layer_scope,
                                     uniform=True)
                w, b, pw, pb, vw, vb = pso_tupple
                self.vweights.append(vw)
                self.vbiases.append(vb)
                self.weights.append(w)
                self.biases.append(b)
                self.pweights.append(pw)
                self.pbiases.append(pb)
                lw = None
                lb = None
                if self.LBPSO:
                    lw = tf.Variable(pw.initialized_value(), name='lbest_w')
                    lb = tf.Variable(pb.initialized_value(), name='lbest_b')
                    self.lbiases.append(lb)
                    self.lweights.append(lw)

                # Multiply by the Velocity Decay
                nextvw = tf.multiply(vw, self.t_VELOCITY_DECAY)
                nextvb = tf.multiply(vb, self.t_VELOCITY_DECAY)

                # Differences between Particle Best & Current
                pdiffw = tf.multiply(tf.subtract(pw, w), pbestrand)
                pdiffb = tf.multiply(tf.subtract(pb, b), pbestrand)
        
                # Differences between the Local Best & Current
                ldiffw = None
                ldiffb = None
                if self.LBPSO:
                    ldiffw = tf.multiply(tf.subtract(lw, w), lbestrand)
                    ldiffb = tf.multiply(tf.subtract(lb, w), lbestrand)

                # Define & Reuse the GBest
                gw = None
                gb = None
                if not self.LBPSO:
                    with tf.variable_scope("gbest", reuse=tf.AUTO_REUSE):
                        gw = tf.get_variable(name='fc' + str(idx + 1) + 'w',
                                            shape=[self.LAYERS[idx], self.LAYERS[idx + 1]],
                                            initializer=tf.zeros_initializer)

                        gb = tf.get_variable(name='fc' + str(idx + 1) + 'b',
                                            shape=[self.LAYERS[idx + 1]],
                                            initializer=tf.zeros_initializer)

                # If first Particle add to Global Else it is already present
                if pno == 0 and not self.LBPSO:
                    self.gweights.append(gw)
                    self.gbiases.append(gb)
                gdiffw = None
                gdiffb = None
                # Differences between Global Best & Current
                if not self.LBPSO:
                    gdiffw = tf.multiply(tf.subtract(gw, w), gbestrand)
                    gdiffb = tf.multiply(tf.subtract(gb, b), gbestrand)
                else:
                    ldiffw = tf.multiply(tf.subtract(lw, w), lbestrand)
                    ldiffb = tf.multiply(tf.subtract(lb, b), lbestrand)

                vweightdiffsum = None
                vbiasdiffsum = None
                if self.LBPSO:
                    vweightdiffsum = tf.multiply(
                        tf.add_n([nextvw, pdiffw, ldiffw]),
                        self.CHI)
                    vbiasdiffsum = tf.multiply(tf.add_n([nextvb, pdiffb, ldiffb]), self.CHI)
                else:
                    vweightdiffsum = tf.add_n([nextvw, pdiffw, gdiffw])
                    vbiasdiffsum = tf.add_n([nextvb, pdiffb, gdiffb])

                vweight_update = None
                if self.VELOCITY_RESTRICT is False:
                    vweight_update = tf.assign(vw, vweightdiffsum, validate_shape=True)
                else:
                    vweight_update = tf.assign(vw, maxclip(vweightdiffsum, self.t_MVEL),
                                            validate_shape=True)

                self.vweight_updates.append(vweight_update)
                vbias_update = None
                if self.VELOCITY_RESTRICT is False:
                    vbias_update = tf.assign(vb, vbiasdiffsum, validate_shape=True)
                else:
                    vbias_update = tf.assign(vb, maxclip(vbiasdiffsum, self.t_MVEL),
                                            validate_shape=True)

                self.vbias_updates.append(vbias_update)
                weight_update = tf.assign(w, w + vw, validate_shape=True)
                self.weight_updates.append(weight_update)
                bias_update = tf.assign(b, b + vb, validate_shape=True)
                self.bias_updates.append(bias_update)

            # Define loss for each of the particle nets
            loss = tf.nn.l2_loss(net - label)
            # loss = tf.reduce_mean(-tf.reduce_sum(label*tf.log(net), reduction_indices=[1]))
            particlebest = tf.cond(loss < pfit, lambda: loss, lambda: pfit)
            fit_update = tf.assign(pfit, particlebest, validate_shape=True)
            self.fit_updates.append(fit_update)
            if not self.LBPSO:
                globalbest = tf.cond(loss < self.gfit, lambda: loss, lambda: self.gfit)
                fit_update = tf.assign(self.gfit, globalbest, validate_shape=True)
                self.fit_updates.append(fit_update)
            control_update = tf.assign(self.t_MVEL, tf.multiply(self.t_MVEL, self.MAX_VEL_DECAY),
                                    validate_shape=True)
            self.control_updates.append(control_update)
            if self.HYBRID:
                optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE)
                hybrid_update = optimizer.minimize(loss)
                self.hybrid_updates.append(hybrid_update)

            # Multiple Length Checks
            assert len(self.weights) == len(self.biases)
            assert len(self.pweights) == len(self.pbiases)
            assert len(self.pweights) == len(self.weights)

            for i in range(len(self.weights)):
                # Particle Best
                pweight = tf.cond(loss <= pfit, lambda: self.weights[
                                i], lambda: self.pweights[i])
                fit_update = tf.assign(self.pweights[i], pweight, validate_shape=True)
                self.fit_updates.append(fit_update)
                pbias = tf.cond(loss <= pfit, lambda: self.biases[i], lambda: self.pbiases[i])
                fit_update = tf.assign(self.pbiases[i], pbias, validate_shape=True)
                self.fit_updates.append(fit_update)

                if self.LBPSO:
                    lneigh = (pno - 1) % self.N_PARTICLES
                    rneigh = (pno + 1) % self.N_PARTICLES
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
                            shape=[self.LAYERS[i], self.LAYERS[i + 1]],
                            name='pbest_w',
                            initializer=tf.random_uniform_initializer)
                        # [self.LAYERS[idx + 1]]
                        lneigh_bias = tf.get_variable(
                            shape=[self.LAYERS[i + 1]],
                            name='pbest_b',
                            initializer=tf.random_uniform_initializer)
                    with tf.variable_scope(rneighscope, reuse=tf.AUTO_REUSE):
                        rneigh_weight = tf.get_variable(
                            shape=[self.LAYERS[i], self.LAYERS[i + 1]],
                            name='pbest_w',
                            initializer=tf.random_uniform_initializer)
                        # [self.LAYERS[idx + 1]]
                        rneigh_bias = tf.get_variable(
                            shape=[self.LAYERS[i + 1]],
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
                    self.fit_updates.append(fit_update)

                    # Deal with Local Best Weights
                    neighbor_best_weight = tf.cond(lfit <= rfit,
                                                lambda: lneigh_weight,
                                                lambda: rneigh_weight)
                    particle_best_weight = tf.cond(pfit <= localfit,
                                                lambda: self.pweights[i],
                                                lambda: self.lweights[i])
                    best_weight = tf.cond(neighbor_best_fit <= particle_best_fit,
                                        lambda: neighbor_best_weight,
                                        lambda: particle_best_weight)
                    fit_update = tf.assign(
                        self.lweights[i], best_weight, validate_shape=True)
                    self.fit_updates.append(fit_update)

                    # Deal with Local Best Biases
                    neighbor_best_bias = tf.cond(lfit <= rfit,
                                                lambda: lneigh_bias,
                                                lambda: rneigh_bias)
                    particle_best_bias = tf.cond(pfit <= localfit,
                                                lambda: self.pbiases[i],
                                                lambda: self.lbiases[i])
                    best_bias = tf.cond(neighbor_best_fit <= particle_best_fit,
                                        lambda: neighbor_best_bias,
                                        lambda: particle_best_bias)
                    fit_update = tf.assign(self.lbiases[i], best_bias, validate_shape=True)
                    self.fit_updates.append(fit_update)

                if not self.LBPSO:
                    # Global Best
                    gweight = tf.cond(loss <= self.gfit,
                                    lambda: self.weights[i],
                                    lambda: self.gweights[i])
                    fit_update = tf.assign(self.gweights[i], gweight, validate_shape=True)
                    self.fit_updates.append(fit_update)
                    gbias = tf.cond(loss <= self.gfit,
                                    lambda: self.biases[i],
                                    lambda: self.gbiases[i])
                    fit_update = tf.assign(self.gbiases[i], gbias, validate_shape=True)
                    self.fit_updates.append(fit_update)

            # Update the lists
            self.nets.append(net)
            self.losses.append(loss)
            print_prog_bar(iteration=pno + 1,
                        total=self.N_PARTICLES,
                        suffix=str_memusage('M'))
        print(self.nets)

        msgtime('Completed\t\t:')

        _var_list = (
            tf.trainable_variables() +
            tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
        # pylint: disable=protected-access
        _var_list += tf.get_collection(tf.GraphKeys._STREAMING_MODEL_PORTS)
        print(len(_var_list))
        for var in _var_list:
            print(var)
        # print(_var_list)

        # Initialize the entire graph
        msgtime('Graph Init Successful\t:')


        '''
        List of all the variables
        for var in tf.global_variables():
            print(var)
        '''

        # Define the updates which are to be done before each iterations
        random_updates = [r.initializer for r in self.random_values]
        updates = self.weight_updates + self.bias_updates + \
            random_updates + self.vbias_updates + self.vweight_updates + \
            self.fit_updates + self.control_updates + self.hybrid_updates
        req_list = None
        if not self.LBPSO:
            req_list = self.losses, updates, self.gfit, self.gbiases, self.vweights, self.vbiases, self.gweights
        else:
            req_list = self.losses, updates, self.vweights, self.vbiases
        return req_list, self.nets