for file in ~/Downloads/2017BOT/non_cancer_subset00/*; do
  echo ${file##*/}
  convert -size 256x256 canvas:black -depth 8 -set colorspace RGB  ${file##*/}.png
done
