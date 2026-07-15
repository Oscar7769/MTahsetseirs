$pdflatex = 'xelatex -synctex=1 -interaction=nonstopmode -no-pdf %O %S && xdvipdfmx -o %B_tmp.pdf %B.xdv && mv -f %B_tmp.pdf %D';
$pdf_mode = 1;
$postscript_mode = $dvi_mode = 0;
