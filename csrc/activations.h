
#ifndef CTIVATIONS_H
#define ACTIVATIONS_H

#define RELU(i) (i) > 0 ? (i) : 0
#define Leaky_RELU(i , a) (i) > 0 ? (i) : ((i) * (a))

#endif //ACTIVATIONS_H