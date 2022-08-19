nz=(64 32 8 4 2 1)
size=(512 1024 2048 4096 8192 16384)
methods=(fourierrec lprec)
dtypes=(float16 float32)
for iz in 5; do
    for im in 1; do
        for id in {0..1}; do
            for ir in 1; do
                rm -rf /local/data_rec/*; 
                sleep 5
                echo ${size[$iz]} ${nz[$iz]} ${methods[$im]} ${dtypes[$id]}
                tomocupy recon --file-name /local/data/tmp${size[$iz]}.h5 --reconstruction-type full --dtype ${dtypes[$id]} --reconstruction-algorithm ${methods[$im]} --max-write-threads 8 --nsino-per-chunk ${nz[$iz]}
                cp time.npy times/time_${size[$iz]}_${nz[$iz]}_${methods[$im]}_${dtypes[$id]}_$ir.npy                
            done
        done
    done
done
