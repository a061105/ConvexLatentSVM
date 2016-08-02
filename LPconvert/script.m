dir = '.'
A = spconvert(load([dir '/A']));
b = load([dir '/b']);
c = load([dir '/c']);
Aeq = spconvert(load([dir '/Aeq']));
beq = load([dir '/beq']);

n = length(c);
lb = zeros(n,1);

options = optimoptions('linprog','Algorithm','interior-point','Display','iter');
[x,fval] = linprog(c,A,b,Aeq,beq,lb,[],[],options);

fp = fopen('sol','w');
fprintf(fp, '%g\n', x);
fclose(fp);

system('paste varMap sol > var_sol');
