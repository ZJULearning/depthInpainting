A=/home/xuehongyang/inpainting/MiddEval3/MiddInpaint

B=$1
C=/home/xuehongyang/inpainting/LRL0_result

Lambda=(30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200)
K=(3 5)
max=30

for var in ${B[@]}; do

    DISPPATH=$A/${var}/disp.png

    MISS=50
    MASK=$A/${var}/mask_${MISS}.png

    echo 'inpainting for' $B 'with mask' ${MASK}
    echo 'initialized with' ${A}/${var}/tnnr_${MISS}.png
    for lam in ${Lambda[@]}; do
        for k in ${K[@]}; do
            paramDir=$C/${lam}_$k
            if [ ! -d "$paramDir" ]; then
                `mkdir ${paramDir}`
            fi

            if [ ! -d "${paramDir}/${var}" ]; then
                `mkdir ${paramDir}/${var}`
            fi
            echo 'lambda_l0 = ' ${lam} ', K = ' $k ', maxCnt = ' ${max}
            OUTPUT=${paramDir}/${var}/lrl0_${MISS}_
            ./depthInpainting LRL0 ${DISPPATH} ${MASK} ${OUTPUT} ${A}/${var}/tnnr_${MISS}.png $k ${lam\
} ${max} ${paramDir}/${var}

        done
    done
done

 
