Afin de compiler toutes les différentes versions il faut se placer dans le dossier [...] et utiliser la
commande :

~/src$ make

Cela permet de compiler les Makefile se trouvant dans les différents fichiers.
La compilation utilise les options suivantes :

-std=c99 -Wall -pg -Wextra -g -Werror -O3 -fopenmp -march=native -I.

Par la suite pour exécuter une version il suffit de se rendre dans le dossier concerné et de
lancer la commande exécutable, par exemple pour la version OpenMP :
Il faut se rendre dans le dossier OpenMP et d'exécuter la commande suivante :

~/src/OpenMP$ ./lanczos_modp --matrix challenge_easy.mtx --prime 65537 --right --n 4 --output kernel.mtx

Chaque version sauvegarde automatiquement tous les 10% d’exécutions. 
Donc pour reprendre depuis la dernière sauvegarde il faut rajouter l’option --c :

~/src/OpenMP$ ./lanczos_modp --matrix challenge_easy.mtx --prime 65537 --right --n 4 --c --output kernel.mtx