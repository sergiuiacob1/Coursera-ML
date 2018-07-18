function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% for i = 1:m
%     hypothesis = sigmoid([1 sigmoid([1 X(i, :)] * Theta1')] * Theta2'); % 1 x 10
%     y_nn = zeros (num_labels, 1); % 10 x 1
%     y_nn (y(i)) = 1;
%     hError = -y_nn' * log (hypothesis') - (1 .- y_nn)' * log (1 .- hypothesis');
%     J = J + 1/m * hError;
% end

X = [ones(m, 1) X]; % m x 401
hypothesis = sigmoid (X * Theta1'); % m x 25
hypothesis = sigmoid ([ones(m, 1) hypothesis] * Theta2'); % m x num_labels

y_nn = zeros (m, num_labels);
for i = 1:m
    y_nn (i, y(i)) = 1;
end

J = 1/m * sum(sum(-y_nn .* log (hypothesis) - (1 - y_nn) .* log (1 - hypothesis)));

theta1Reg = Theta1(:, 2:size(Theta1, 2));
theta2Reg = Theta2(:, 2:size(Theta2, 2));
reg = lambda * (sum(sum(theta1Reg .^ 2)) + sum(sum(theta2Reg .^ 2))) / (2 * m);
J += reg;

% Theta1 has size 25 x 401
% Theta2 has size 10 x 26

for t = 1:m
    % Feedforward
    % Step 1
    % X already has the bias term
    a1 = X(t, :)'; % 401 x 1
    z2 = Theta1 * a1; % 25 x 1
    a2 = sigmoid (z2);
    a2 = [1; a2]; % 26 x 1
    z3 = Theta2 * a2; % 10 x 1
    a3 = sigmoid (z3); % my hypothesis, 10 x 1

    % Backpropagation

    % Step 2
    delta3 = a3 - y_nn (t, :)'; % 10 x 1
    z2 = [1; z2]; % add bias term
    % Step 3
    delta2 = (Theta2' * delta3) .* sigmoidGradient (z2);
    % Step 4
    delta2 = delta2 (2:end); % skip ∂(2, 0);
    Theta2_grad += delta3 * a2';
    Theta1_grad += delta2 * a1';
end;

% Step 5
Theta2_grad /= m;
Theta1_grad /= m;







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
