function [weight_v] = TKA_kernels_weights(Kernels_list,adjmat,dim)

% adjmat : binary adjacency matrix
% dim    : dimension (1 - rows, 2 - cols)

% 论文 Predicting Protein-DNA Binding Residues by Weightedly Combining Sequence-Based Features and Boosting Multiple SVMs 讲CLKTA


num_kernels = size(Kernels_list,3);

%Knormalized Kernels

%for i=1:num_kernels
	
%	S=Knormalized(Kernels_list(:,:,i));
%	Kernels_list(:,:,i) = S;

%end

weight_v = zeros(num_kernels,1);

y = adjmat;
    % Graph based kernel
if dim == 1
        ga = y*y';
else
        ga = y'*y;
end
Alignment_list=[];
%ga=Knormalized(ga);
for i=1:num_kernels
	
	A_v = kernel_alignment(Kernels_list(:,:,i),ga);
	Alignment_list = [Alignment_list;A_v];

end

for i=1:num_kernels
	
	weight_v(i) = Alignment_list(i)/(sum(Alignment_list));

end

end

%kernel alignment
function A = kernel_alignment(K1,ideal_k)
	   
	   v_1 = trace(K1'*ideal_k); % < , >  的F范数  求矩阵对角线 元素的和
	   v_n = size(K1,1);
	   v_2 = sqrt(trace(K1'*K1));    %  || || 的F范数
	   
	   A = v_1/(v_n*v_2);  % CLKTA �? 权重


end