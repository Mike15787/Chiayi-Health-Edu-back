from databases import SessionLocal, AgentSettings

# 欄位說明
# agent_code 表示目前難度
# med_code 表示是否有特殊用藥

# 在特殊設定(special_status)包含 "不知道檢查型態" 目前check_type 會需要要再加入"不知道" 
# 新增欄位 繳交費用 800/4500

# 新增欄位 過去有沒有用過瀉藥經驗

data_list = [
    {
        "agent_code": "A1",
        "gender": "女性",
        "age": "57歲",
        "med_info": "Mopride F.C 5mg tab (順買錠)1粒 BID PO (N)",
        "disease": "消化不良、心房纖維顫動",
        "med_complexity": "低(<5項)",
        "med_code": "抗凝血劑 S1",
        "special_status": "無",
        "check_day": 10,
        "check_time": "上午08:10",
        "check_type": "無痛檢查",
        "low_cost_med": "無",
        "payment_fee": "4500", 
        "laxative_experience": "無"
    },
    {
        "agent_code": "A2",
        "gender": "女性",
        "age": "78歲",
        "med_info": "Stazyme F.C.tab(速泰消)1粒 TID PO(N)\nEliquis 5mg(艾必克凝)0.5粒 BID PO(S1)\nAtivan 1mg tab(悠然)1粒 HS PO(N)\nErispan 0.25mg tab(易舒)1粒 TID PO(N)\nXanax 0.5mg tab(賽安諾)1粒 BID PO(N)\nCymbalta 30mg cap(勃景停)1粒 HS PO(N)",
        "disease": "廣泛性焦慮症、其他睡眠障礙症、心悸",
        "med_complexity": "低(<5項)",
        "med_code": "無特殊用藥N",
        "special_status": "情緒問題",
        "check_day": 3,
        "check_time": "上午10:00",
        "check_type": "無痛檢查",
        "low_cost_med": "無",
        "payment_fee": "4500", 
        "laxative_experience": "無"
    },
    {
        "agent_code": "A3",
        "gender": "男性",
        "age": "55歲",
        "med_info": "1.25mg Concor tab(康肯)1粒 QD PO(N)\n75mg Plavix tab(保栓通)1粒 QD PO(S2)\nImdur 60mg tab(冠欣)1粒QD PO(N)\nEntresto 100mg tab(健安心)1粒BID PO(N)",
        "disease": "自體的冠狀動脈粥樣硬化心臟病伴有其他形式之心絞痛、慢性心臟衰竭",
        "med_complexity": "低(<5項)",
        "med_code": "抗血小板藥物 S2",
        "special_status": "不知道檢查型態",
        "check_day": 7,
        "check_time": "上午11:20",
        "check_type": "無痛檢查",
        "low_cost_med": "無",
        "payment_fee": "4500", 
        "laxative_experience": "無"
    },
    {
        "agent_code": "A4",
        "gender": "女性",
        "age": "72歲",
        "med_info": "Adalat-OROS 30mg tab(恆脈循)2粒 BID PO(X1)\nConslife tab (秘福)2粒HS PO(S4)",
        "disease": "高血壓、便祕",
        "med_complexity": "低(<5項)",
        "med_code": "高血壓藥物 X1, 緩解便秘用藥 S4",
        "special_status": "無",
        "check_day": 5,
        "check_time": "上午12:00",
        "check_type": "一般檢查",
        "low_cost_med": "無",
        "payment_fee": "800", 
        "laxative_experience": "有"
    },
    {
        "agent_code": "A5",
        "gender": "男性",
        "age": "75歲",
        "med_info": "Pantoloc (保衛康治潰樂)1粒QD PO(N)\nActoS 15mg tab(愛妥糖)1粒 QD PO(S3)\nGlucophage 500mg tab(伏糖)1粒 QD PO(S3)",
        "disease": "慢性胃潰瘍、第二型糖尿病",
        "med_complexity": "低(<5項)",
        "med_code": "降血糖藥 S3",
        "special_status": "無",
        "check_day": 14,
        "check_time": "下午14:30",
        "check_type": "一般檢查",
        "low_cost_med": "無",
        "payment_fee": "800", 
        "laxative_experience": "無"
    },
    {
        "agent_code": "B1",
        "gender": "女性",
        "age": "42歲",
        "med_info": "Valdoxan(煩多閃)1粒HS PO(N)\nImoVAne(樂比克)1粒HS PO(N)\nEpRam(抑鬱)1粒HS PO(N)\nXanax(贊安諾)1粒QDHS PO(N)\nAlinamin-F(合利他命)1粒QD PO(N)\nFemara(復乳納)1粒QD PO(N)",
        "disease": "廣泛性焦慮症、乳房惡性腫瘤",
        "med_complexity": "中(5-9項)",
        "med_code": "無特殊用藥 N",
        "special_status": "無",
        "check_day": 10,
        "check_time": "上午11:20",
        "check_type": "無痛檢查",
        "low_cost_med": "無",
        "payment_fee": "4500", 
        "laxative_experience": "無"
    },
    {
        "agent_code": "B2",
        "gender": "男性",
        "age": "64歲",
        "med_info": "Exforge(易安穩)1粒BID PO(X1)\nDilatrend(心全)1粒BID PO(X1)\nLipanthyl(弗尼利脂寧)1粒QD PO(N)\nAtozet (優泰脂)1粒QD PO(N)\nHarnalidge-OCAS(活路利淨)1粒HS PO(N)\nPlavix(保栓通)1粒QD PO(S2)\nBokey(安心平)1粒QD PO(S2)",
        "disease": "高血壓、其他高血脂症、攝護腺腫大伴有下泌尿道症狀、高血壓性心臟病、心肌梗塞",
        "med_complexity": "中(5-9項)",
        "med_code": "抗血小板藥物 S2; 高血壓藥物 X1",
        "special_status": "不知道檢查型態",
        "check_day": 3,
        "check_time": "上午12:00",
        "check_type": "無痛檢查",
        "low_cost_med": "無",
        "payment_fee": "4500", 
        "laxative_experience": "無"
    },
    {
        "agent_code": "B3",
        "gender": "男性",
        "age": "34歲",
        "med_info": "Amaryl 2mg tab(瑪爾胰)0.5粒 QD PO(S3)\nJARDiance 10mg tab(恩排糖)1粒 QD PO(S3)\nNovo-Norm 1mg tab(醣立定)1粒 BIDM PO(S3)\nLiVAlo 2mg tab(力清之)1粒 QD PO(N)\nLinicor tab(理脂)1粒 QD PO(N)\nFeBuric 80mg tab(福避痛)0.5粒QD PO(N)\nSmecta 3g powder(舒腹達)1包 QD PO(N)",
        "disease": "第二型糖尿病、混合型高血脂症、慢性痛風，未伴有痛風石、功能性腹瀉",
        "med_complexity": "中(5-9項)",
        "med_code": "降血糖藥 S3",
        "special_status": "無",
        "check_day": 7,
        "check_time": "下午14:30",
        "check_type": "一般檢查",
        "low_cost_med": "無",
        "payment_fee": "800", 
        "laxative_experience": "無"
    },
    {
        "agent_code": "B4",
        "gender": "男性",
        "age": "52歲",
        "med_info": "Conslife tab(秘福)2粒BID PO(S4)\nGasCON 40mg tab(加斯克兒)2粒 QID PO(N)\nTakepron OD 15mg tab(泰克胃通)2粒QD PO(N)\nCordarone 200mg tab(臟得樂)0.5粒QD PO(N)\n1mg Coumadin tab(可化凝)0.5粒QD PO(S1)",
        "disease": "胃及十二指腸息肉、便秘、心房纖維顫動",
        "med_complexity": "中(5-9項)",
        "med_code": "緩解便秘用藥 S4; 抗凝血劑 S1",
        "special_status": "無",
        "check_day": 5,
        "check_time": "上午08:10",
        "check_type": "一般檢查",
        "low_cost_med": "無",
        "payment_fee": "800", 
        "laxative_experience": "有"
    },
    {
        "agent_code": "B5",
        "gender": "女性",
        "age": "78歲",
        "med_info": "CeRenin 40mg tab(血循)1粒 TID PO(N)\nAlinamin-F tab(合利他命)1粒BID PO(N)\nArcoxia 60mg tab(萬克適)1粒 QD PO(N)\nBolaxin 500mg tab(寶樂欣)1粒TID PO(N)\nLixiana 60mg tab(里先安)0.5粒 QD PO(S1)\nDepakine syrup 40cc(帝拔癲口服液)2CC BID PO(X2)\n25mg Seroquel tab(東健)1粒 PRHS PO(N)",
        "disease": "診斷欠明之腦血管疾病、未明示頸椎之頸椎間盤疾患伴有脊髓病變",
        "med_complexity": "中(5-9項)",
        "med_code": "抗凝血劑 S1; 抗癲癇藥物 X2",
        "special_status": "聽不懂國語",
        "check_day": 14,
        "check_time": "上午10:00",
        "check_type": "無痛檢查",
        "low_cost_med": "無",
        "payment_fee": "4500", 
        "laxative_experience": "無"
    },
    {
        "agent_code": "C1",
        "gender": "女性",
        "age": "65歲",
        "med_info": "Sevikar 5/40mg tab(舒脈康)1粒QD PO(X1)\nNorvasc 5mg tab(脈優)1粒QN PO(X1)\nTrental SR 400mg tab(循妥斯)1粒BID PO(N)\n(自)Forxiga 10mg tab(福適佳)1粒QD PO(S3)\nLescol XL 80mg tab(益脂可)1粒HS PO(N)\nSalazopyrin-EN 500mg tab(撒樂)1粒QDPO(N)\nFA 5mg tab(葉酸)1粒QDPO(N)\nAllopurinol 100mg tab(威寧疼)1粒QD PO(N)\nUrecholine 25mg tab(滯尿通)1粒BID PO(N)\nOxbu 5mg tab(歐舒)1粒QD PO(N)\nBetmiga 50mg tab(貝坦利)1粒QD PO(N)\nDormicum 7.5mg tab(導美睡)1粒HS PO(N)",
        "disease": "慢性腎臟病(stage 3b)、蛋白尿、高血壓、純高膽固醇血症、潰病性結腸炎、痛風性關節炎、膀胱過動症、睡眠障礙",
        "med_complexity": "高(≥10項)",
        "med_code": "高血壓藥物 X1; 降血糖藥 S3",
        "special_status": "無",
        "check_day": 10,
        "check_time": "上午12:00",
        "check_type": "無痛檢查",
        "low_cost_med": "無",
        "payment_fee": "4500", 
        "laxative_experience": "無"
    },
    {
        "agent_code": "C2",
        "gender": "男性",
        "age": "72歲",
        "med_info": "Sigmart 5mg tab(利可心)1粒BID PO(N)\nBokey 100mg cap(安心平)1粒QD PO(S2)\nMexitil 100mg cap(脈律循)1粒BID PO(N)\nLiVAlo 2mg tab(力清之)1粒QD PO(N)\nMadopar 250mg tab(美道普)1粒BID PO(N)\nArtane 2mg tab(瑞丹)0.5粒QD PO(N)\nFosamax plus 70mg/5600IU tab(福善美保骨)1粒QW4 PO(N)\nProcal 667mg tab(普羅鈣)1粒BID PO(N)\nTraceton tab(服安痛)1粒PRQID PO(N)\nARicept 10mg tab(愛憶欣)1粒HS PO(N)\nHydergine 1.5mg tab(益利循)1粒BID PO(N)\nSpiolto Respimat(適倍樂)2噴QD INHL(N)\nFluimucil 600mg tab(愛克痰)1粒BID PO(N)\nUrief 8mg tab(優列扶)0.5粒QN PO(N)",
        "disease": "心絞痛、心臟節律不整、高血脂症、帕金森氏病、未明示部位之老年性骨質疏鬆症伴有病理性骨折之初期照護、阿茲海默氏病、慢性阻塞性肺病、攝護腺增大未伴有下泌尿道症狀",
        "med_complexity": "高(≥10項)",
        "med_code": "抗血小板藥物 S2",
        "special_status": "講台語",
        "check_day": 3,
        "check_time": "下午14:30",
        "check_type": "無痛檢查",
        "low_cost_med": "無",
        "payment_fee": "4500", 
        "laxative_experience": "無"
    },
    {
        "agent_code": "C3",
        "gender": "男性",
        "age": "61歲",
        "med_info": "Diamicron MR 60mg tab(岱蜜克龍)1粒BID PO(S3)\nQtern 5/10mg tab(控糖穩)1粒QD PO(S3)\nAcarbose 50mg tab(抵克醣)1粒BIDM PO(S3)\nGlucophage 500mg tab(伏糖)1粒BID PO(S3)\nKoBal 500mcg cap(Methycobal)1粒TID PO(N)\nXarelto 10mg FC tab(拜瑞妥)1粒QD PO(S1)\nLinicor tab(理脂)1粒QD PO(N)\nLipanthyl(弗尼利脂寧)1粒QD PO(N)\nEfient F.C. 3.75 mg tab(抑凝安)1粒QD PO(S2)\nBokey(安心平)1粒QD PO(S2)\nGaster 20mg tab(非潰)1粒QD PO(N)",
        "disease": "第二型糖尿病，伴有糖尿病的神經病變、髖關節裝置物植入術後療養、混合型高血脂症、非ST段上升之心肌梗塞",
        "med_complexity": "高(≥10項)",
        "med_code": "降血糖藥 S3; 抗凝血劑 S1; 抗血小板藥物 S2",
        "special_status": "無",
        "check_day": 7,
        "check_time": "上午10:00",
        "check_type": "一般檢查",
        "low_cost_med": "無",
        "payment_fee": "800", 
        "laxative_experience": "無"
    },
    {
        "agent_code": "C4",
        "gender": "男性",
        "age": "83歲",
        "med_info": "Xeloda 500mg tab(截瘤達)2粒BID PO(N)\nPanagesic 30mg tab(保樂健)1粒BID PO(N)\nTramal 50mg cap(頓痛特)2粒QID PO(N)\nBerotec N 100mcg(備勞喘)1噴PRN INHL(N)\nTrelegy Ellipta 92/55/22 mcg(肺樂喜)1噴QD INHL(N)\nFlucil 200mg pack(愛克痰)1包TID PO(N)\nTwynsta 80/5mg tab(倍必康平)1粒QD PO(X1)\nCaduet 5mg/10mg tab(脂脈優)1粒QN PO(X1)\nDulcolax 5mg tab(樂可舒)2粒TID PO(S4)\nDulcolax 10mg supp(無秘)1粒HS SUPP(S4)",
        "disease": "縱膈之續發性惡性腫瘤、慢性阻塞性肺病、本態性(原發性)高血壓、其他便秘",
        "med_complexity": "高(≥10項)",
        "med_code": "緩解便秘用藥 S4; 高血壓藥物 X1",
        "special_status": "聽不懂國語",
        "check_day": 5,
        "check_time": "上午08:10",
        "check_type": "一般檢查",
        "low_cost_med": "無",
        "payment_fee": "800", 
        "laxative_experience": "有"
    },
    {
        "agent_code": "C5",
        "gender": "女性",
        "age": "88歲",
        "med_info": "Allopurinol 100mg tab(優力康)1粒BID PO(N)\nColchicine 0.5mg tab(秋水仙鹼)1粒QOD PO(N)\nAPAP 500mg tab(伯樂止痛)1粒TID PO(N)\nEvista 60mg tab(鈣穩)1粒QD PO(N)\nBiocal Plus tab(滋骨加強)1粒BID PO(N)\n(膏)Voren-G gel 40gm(非炎)PRN TOPI(N)\nKary Uni 5mL oph susp(柯寧優尼)1滴TID OU(N)\nDuratears 3.5g eye ointment (淚膜)1次HS OU(N)\nDexilant 60mg cap(得喜胃通)1粒QD PO(N)\nMopride F.C 5mg tab(順胃暢)1粒BID PO(N)",
        "disease": "未明示部位特發性慢性痛風，伴有痛風石、老年性骨質疏鬆症未伴有病理性骨折、白內障、乾眼症、慢性十二指腸潰瘍，未伴有出血或穿孔",
        "med_complexity": "高(≥10項)",
        "med_code": "非/無特殊用藥 N",
        "special_status": "不知道檢查型態",
        "check_day": 14,
        "check_time": "上午11:20",
        "check_type": "一般檢查",
        "low_cost_med": "無",
        "payment_fee": "800", 
        "laxative_experience": "無"
    }
]

def insert_agent_data():
    """將病例資料插入到資料庫"""
    db = SessionLocal()
    try:
        # 先檢查是否已存在資料，避免重複插入
        existing_agents = db.query(AgentSettings).all()
        existing_codes = [agent.agent_code for agent in existing_agents]
        
        inserted_count = 0
        for item in data_list:
            # --- 新增：資料預處理邏輯 ---
            if "不知道檢查型態" in item.get("special_status", ""):
                item["check_type"] = "不知道"
            # --- 結束 ---
            
            if item["agent_code"] not in existing_codes:
                agent = AgentSettings(**item)
                db.add(agent)
                inserted_count += 1
                print(f"插入病例: {item['agent_code']} - {item['gender']} {item['age']}")
            else:
                print(f"病例 {item['agent_code']} 已存在，跳過插入")
        
        db.commit()
        print(f"\n成功插入 {inserted_count} 筆病例資料！")
        print(f"總計資料庫中有 {len(existing_agents) + inserted_count} 筆病例資料")
        
    except Exception as e:
        print(f"插入資料時發生錯誤: {e}")
        db.rollback()
    finally:
        db.close()

def show_all_agents():
    """顯示所有病例資料"""
    db = SessionLocal()
    try:
        agents = db.query(AgentSettings).all()
        print(f"\n=== 資料庫中的所有病例資料 (共 {len(agents)} 筆) ===")
        for agent in agents:
            print(f"{agent.agent_code}: {agent.gender} {agent.age} - 疾病: {agent.disease} | 檢查類型: {agent.check_type} | 費用: {agent.payment_fee} | 瀉藥經驗: {agent.laxative_experience}")
    except Exception as e:
        print(f"查詢資料時發生錯誤: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    print("開始插入病例資料...")
    insert_agent_data()
    
    print("\n" + "="*50)
    show_all_agents()