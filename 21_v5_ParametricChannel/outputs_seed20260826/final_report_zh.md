# 实验21 v5——低维因果参数化响应实验

本报告仅覆盖随机种子 20260826，不能表述为跨随机种子稳定性。

## 五折结果

|condition|test_fold|auroc|auprc|balanced_accuracy|f1|sensitivity|specificity|threshold|
|---|---|---|---|---|---|---|---|---|
|B_parametric_fixed_first|0|0.8823|0.8972|0.8112|0.8006|0.7581|0.8642|0.6453|
|C_parametric_unit_energy|0|0.8852|0.8984|0.8105|0.7979|0.7482|0.8727|0.5582|
|D_parametric_no_physics|0|0.8789|0.8928|0.8083|0.8003|0.7680|0.8487|0.4647|
|B_parametric_fixed_first|1|0.9165|0.9303|0.8426|0.8340|0.7906|0.8946|0.6629|
|C_parametric_unit_energy|1|0.9111|0.9225|0.8389|0.8361|0.8214|0.8565|0.3717|
|D_parametric_no_physics|1|0.9085|0.9272|0.8448|0.8392|0.8097|0.8799|0.5178|
|B_parametric_fixed_first|2|0.9183|0.9253|0.8522|0.8457|0.8103|0.8941|0.4200|
|C_parametric_unit_energy|2|0.9302|0.9371|0.8647|0.8593|0.8265|0.9029|0.4350|
|D_parametric_no_physics|2|0.9185|0.9244|0.8471|0.8434|0.8235|0.8706|0.7293|
|B_parametric_fixed_first|3|0.9281|0.9288|0.8523|0.8488|0.8294|0.8752|0.5685|
|C_parametric_unit_energy|3|0.9200|0.9250|0.8419|0.8319|0.7822|0.9015|0.5127|
|D_parametric_no_physics|3|0.9243|0.9313|0.8495|0.8460|0.8266|0.8724|0.3088|
|B_parametric_fixed_first|4|0.8603|0.8586|0.7715|0.7500|0.6855|0.8575|0.5208|
|C_parametric_unit_energy|4|0.8428|0.8341|0.7673|0.7650|0.7574|0.7772|0.6445|
|D_parametric_no_physics|4|0.8404|0.8404|0.7616|0.7582|0.7475|0.7757|0.4518|

## 合并折外预测

|condition|auroc|auprc|balanced_accuracy|f1|sensitivity|specificity|threshold|
|---|---|---|---|---|---|---|---|
|A_v4_source_only|0.8812|0.8873|0.8160|0.8040|0.7546|0.8774|0.6954|
|B_parametric_fixed_first|0.8949|0.9054|0.8230|0.8089|0.7494|0.8966|0.6453|
|C_parametric_unit_energy|0.8746|0.8689|0.8220|0.8122|0.7700|0.8740|0.5582|
|D_parametric_no_physics|0.8894|0.8940|0.8177|0.8134|0.7946|0.8409|0.4647|

## 原始录音配对 Bootstrap（2000 次）

|comparison|metric|estimate|ci_low|ci_high|runs|
|---|---|---|---|---|---|
|C_minus_A|auroc|-0.0066|-0.0201|0.0061|2000|
|C_minus_A|auprc|-0.0184|-0.0421|0.0074|2000|
|C_minus_A|balanced_accuracy|0.0060|-0.0014|0.0553|2000|
|C_minus_A|f1|0.0083|-0.0084|0.1000|2000|
|C_minus_A|sensitivity|0.0154|-0.0571|0.2097|2000|
|C_minus_A|specificity|-0.0034|-0.1100|0.0765|2000|
|C_minus_B|auroc|-0.0203|-0.0318|-0.0083|2000|
|C_minus_B|auprc|-0.0364|-0.0557|-0.0129|2000|
|C_minus_B|balanced_accuracy|-0.0010|-0.0124|0.0082|2000|
|C_minus_B|f1|0.0033|-0.0217|0.0222|2000|
|C_minus_B|sensitivity|0.0206|-0.0614|0.1013|2000|
|C_minus_B|specificity|-0.0226|-0.1013|0.0461|2000|
|C_minus_D|auroc|-0.0148|-0.0278|-0.0024|2000|
|C_minus_D|auprc|-0.0251|-0.0464|-0.0024|2000|
|C_minus_D|balanced_accuracy|0.0043|-0.0061|0.0252|2000|
|C_minus_D|f1|-0.0012|-0.0172|0.0478|2000|
|C_minus_D|sensitivity|-0.0246|-0.0732|0.1314|2000|
|C_minus_D|specificity|0.0331|-0.0927|0.1052|2000|

## 每 5 dB SNR 区间

|condition|snr_db|n_positive|auroc|auprc|sensitivity|
|---|---|---|---|---|---|
|A_v4_source_only|[-15,-10)|566|0.7222|0.3605|0.4170|
|A_v4_source_only|[-10,-5)|604|0.8110|0.5077|0.5762|
|A_v4_source_only|[-5,0)|589|0.9053|0.6764|0.7963|
|A_v4_source_only|[0,5)|585|0.9388|0.7390|0.8769|
|A_v4_source_only|[5,10)|574|0.9507|0.7602|0.9164|
|A_v4_source_only|[10,15]|582|0.9582|0.7768|0.9433|
|B_parametric_fixed_first|[-15,-10)|566|0.7406|0.3677|0.4117|
|B_parametric_fixed_first|[-10,-5)|604|0.8311|0.5519|0.5861|
|B_parametric_fixed_first|[-5,0)|589|0.9149|0.7175|0.7742|
|B_parametric_fixed_first|[0,5)|585|0.9487|0.8043|0.8718|
|B_parametric_fixed_first|[5,10)|574|0.9620|0.8365|0.9094|
|B_parametric_fixed_first|[10,15]|582|0.9708|0.8573|0.9416|
|C_parametric_unit_energy|[-15,-10)|566|0.7175|0.3242|0.4523|
|C_parametric_unit_energy|[-10,-5)|604|0.8062|0.4566|0.5927|
|C_parametric_unit_energy|[-5,0)|589|0.8933|0.5820|0.8014|
|C_parametric_unit_energy|[0,5)|585|0.9322|0.6714|0.8957|
|C_parametric_unit_energy|[5,10)|574|0.9453|0.6890|0.9338|
|C_parametric_unit_energy|[10,15]|582|0.9521|0.7076|0.9433|
|D_parametric_no_physics|[-15,-10)|566|0.7398|0.3791|0.4700|
|D_parametric_no_physics|[-10,-5)|604|0.8280|0.5239|0.6540|
|D_parametric_no_physics|[-5,0)|589|0.9099|0.6805|0.8353|
|D_parametric_no_physics|[0,5)|585|0.9349|0.7450|0.8855|
|D_parametric_no_physics|[5,10)|574|0.9572|0.7823|0.9460|
|D_parametric_no_physics|[10,15]|582|0.9652|0.8020|0.9742|

## h 与参数诊断

|condition|parameter|mean|median|boundary_fraction|
|---|---|---|---|---|
|B_parametric_fixed_first|d_ms|0.9318|0.9313|0.0491|
|B_parametric_fixed_first|tau_ms|0.1259|0.1073|0.0000|
|B_parametric_fixed_first|amplitude|-0.0088|-0.0087|0.0000|
|C_parametric_unit_energy|d_ms|0.8855|0.7869|0.0000|
|C_parametric_unit_energy|tau_ms|0.1266|0.1106|0.0000|
|C_parametric_unit_energy|amplitude|-0.0098|-0.0067|0.0000|
|D_parametric_no_physics|d_ms|0.9391|1.0996|0.0000|
|D_parametric_no_physics|tau_ms|0.1490|0.1283|0.0000|
|D_parametric_no_physics|amplitude|-0.0133|-0.0086|0.0000|

A 是只读复用的 v4 source_only。C−B 检查 L2 归一化，C−D 检查晚到能量项，C−A 检查整体变化；A−B 不作为纯架构比较。负样本 h 仅作单独展示，不作为海豚传播信道证据。若 h 接近单位冲激或交换诊断未改善，不能以曲线丰富替代机制证据。
