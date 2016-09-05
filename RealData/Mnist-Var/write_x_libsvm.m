function write_x_libsvm( fp, x, truncate_thd )

x(abs(x)<truncate_thd)=0;

[r,c,v] = find(x);
fprintf(fp, '%d:%g ', [c;v]);
