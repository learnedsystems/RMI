rm -f ../competitors/rmi/all_rmis.h

for header in $(ls ../competitors/rmi/nm*.h); do
    echo "#include \"`basename ${header}`\"" >> ../competitors/rmi/all_rmis.h
done
