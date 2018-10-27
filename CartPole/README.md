# CartPole-v1

Balancing of an un-actuated pole on a cart.


### cartPole_PID.py
Balancing is achieved using PD-control to acquire the desired cart velocity given the pole angle and (angular) velocity. Bang-bang control (since possible controls are binary) is used to decide on the cart steering (i.e. which way to push the cart) given the current and desired velocity.


Currently, P- and D- parameters are tuned manually (since they've worked well from the beginning). I have plans to make them learnable in the future.
