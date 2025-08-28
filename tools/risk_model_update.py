# 创业公司风险预测模型 - 改进版本
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import shap

warnings.filterwarnings('ignore')

class StartupRiskPredictor:
    def __init__(self, data_path):
        """
        初始化预测器
        """
        self.data_path = data_path
        self.df_raw = None
        self.df_final = None
        self.models = {}
        self.feature_names = None
        
    def load_and_explore_data(self):
        """
        1. 数据加载和探索
        """
        print("=" * 50)
        print("📊 数据加载和探索")
        print("=" * 50)
        
        try:
            self.df_raw = pd.read_csv(self.data_path)
            print(f"✅ 成功加载数据，形状: {self.df_raw.shape}")
            
            # 缺失值统计
            missing = self.df_raw.isnull().sum()
            missing_stats = missing[missing > 0].sort_values(ascending=False)
            
            if len(missing_stats) > 0:
                print("\n📋 缺失值统计:")
                for col, count in missing_stats.items():
                    pct = count / len(self.df_raw) * 100
                    print(f"  {col}: {count} ({pct:.1f}%)")
            else:
                print("✅ 无缺失值")
                
            return self.df_raw.head()
            
        except FileNotFoundError:
            print(f"❌ 文件未找到: {self.data_path}")
            return None
        except Exception as e:
            print(f"❌ 数据加载失败: {str(e)}")
            return None
    
    def clean_data(self):
        """
        2. 数据清洗
        """
        print("\n" + "=" * 50)
        print("🧹 数据清洗")
        print("=" * 50)
        
        if self.df_raw is None:
            print("❌ 请先加载数据")
            return None
            
        # 删除关键字段缺失的行
        critical_columns = ['founded_at', 'category_list', 'country_code']
        initial_rows = len(self.df_raw)
        
        df_clean = self.df_raw.dropna(subset=critical_columns)
        
        print(f"原始行数: {initial_rows}")
        print(f"清洗后行数: {len(df_clean)} (保留 {len(df_clean)/initial_rows*100:.1f}%)")
        
        # 目标变量分析
        if 'status' in df_clean.columns:
            status_counts = df_clean['status'].value_counts(dropna=False)
            print(f"\n📈 公司状态分布:")
            for status, count in status_counts.items():
                pct = count / len(df_clean) * 100
                print(f"  {status}: {count} ({pct:.1f}%)")
        
        self.df_clean = df_clean
        return df_clean
    
    def feature_engineering(self):
        """
        3. 特征工程
        """
        print("\n" + "=" * 50)
        print("🔧 特征工程")
        print("=" * 50)
        
        if not hasattr(self, 'df_clean'):
            print("❌ 请先进行数据清洗")
            return None
            
        df_model = self.df_clean.copy()
        
        # 创建目标变量
        df_model['is_risky'] = (df_model['status'] == 'closed').astype(int)
        print(f"风险样本比例: {df_model['is_risky'].mean():.3f}")
        
        # 提取成立年份
        df_model['founded_year'] = pd.to_datetime(
            df_model['founded_at'], 
            errors='coerce'
        ).dt.year
        
        # 清理融资金额
        if 'funding_total_usd' in df_model.columns:
            df_model['funding_total_usd'] = (
                df_model['funding_total_usd']
                .astype(str)
                .str.replace(r'[\$,]', '', regex=True)
                .replace(['', '-', 'nan'], np.nan)
                .astype(float)
            )
            
            # 用中位数填充缺失值
            median_funding = df_model['funding_total_usd'].median()
            df_model['funding_total_usd'] = df_model['funding_total_usd'].fillna(median_funding)
            print(f"融资金额中位数填充: ${median_funding:,.0f}")
        
        # 处理分类特征
        # 简化行业类别
        df_model['main_category'] = df_model['category_list'].apply(
            lambda x: str(x).split('|')[0] if '|' in str(x) else str(x)
        )
        
        # 简化国家
        top_countries = df_model['country_code'].value_counts().head(10).index
        df_model['country_grouped'] = df_model['country_code'].apply(
            lambda x: x if x in top_countries else 'Other'
        )
        
        # One-Hot编码
        categorical_features = ['country_grouped', 'main_category']
        df_encoded = pd.get_dummies(
            df_model[categorical_features], 
            prefix=['country', 'category']
        )
        
        # 选择数值特征
        numeric_features = ['founded_year', 'funding_total_usd', 'funding_rounds']
        available_numeric = [col for col in numeric_features if col in df_model.columns]
        
        # 合并特征
        self.df_final = pd.concat([
            df_model[available_numeric + ['is_risky']],
            df_encoded
        ], axis=1)
        
        # 处理剩余缺失值
        numeric_cols = self.df_final.select_dtypes(include=[np.number]).columns
        numeric_cols = numeric_cols.drop('is_risky')
        
        for col in numeric_cols:
            if self.df_final[col].isnull().any():
                fill_value = self.df_final[col].median() if col != 'founded_year' else 2010
                self.df_final[col] = self.df_final[col].fillna(fill_value)
                print(f"填充 {col} 缺失值: {fill_value}")
        
        print(f"✅ 最终特征维度: {self.df_final.shape}")
        print(f"✅ 无缺失值: {not self.df_final.isnull().any().any()}")
        
        return self.df_final
    
    def prepare_train_test_split(self, test_size=0.2, random_state=42):
        """
        4. 训练测试集划分
        """
        from sklearn.model_selection import train_test_split
        
        if self.df_final is None:
            print("❌ 请先进行特征工程")
            return None
            
        X = self.df_final.drop("is_risky", axis=1)
        y = self.df_final["is_risky"]
        
        self.feature_names = X.columns.tolist()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\n📊 数据集划分:")
        print(f"训练集: {len(self.X_train)} 样本")
        print(f"测试集: {len(self.X_test)} 样本")
        print(f"训练集风险比例: {self.y_train.mean():.3f}")
        print(f"测试集风险比例: {self.y_test.mean():.3f}")
        
    def train_models(self):
        """
        5. 模型训练 - 使用多种算法
        """
        print("\n" + "=" * 50)
        print("🤖 模型训练")
        print("=" * 50)

        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        from sklearn.metrics import classification_report, roc_auc_score
        import joblib  # ✅ 加载 joblib

        try:
            from xgboost import XGBClassifier
            xgb_available = True
        except ImportError:
            print("⚠️ XGBoost未安装，将跳过XGB模型")
            xgb_available = False

        models = {
            'LogisticRegression': LogisticRegression(
                max_iter=1000, 
                class_weight='balanced', 
                random_state=42
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100, 
                class_weight='balanced', 
                random_state=42
            )
        }

        if xgb_available:
            models['XGBoost'] = XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=10,
                random_state=42
            )

        self.models = {}
        model_scores = {}

        for name, model in models.items():
            print(f"\n🔄 训练 {name}...")
            try:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_proba = model.predict_proba(self.X_test)[:, 1]

                auc_score = roc_auc_score(self.y_test, y_proba)
                model_scores[name] = auc_score
                self.models[name] = model

                print(f"✅ {name} AUC: {auc_score:.3f}")

            except Exception as e:
                print(f"❌ {name} 训练失败: {str(e)}")

        if len(self.models) > 1:
            print(f"\n🔄 训练集成模型...")
            ensemble_models = [(name, model) for name, model in self.models.items()]
            voting_clf = VotingClassifier(
                estimators=ensemble_models,
                voting='soft'
            )

            voting_clf.fit(self.X_train, self.y_train)
            y_proba_ensemble = voting_clf.predict_proba(self.X_test)[:, 1]
            auc_ensemble = roc_auc_score(self.y_test, y_proba_ensemble)

            self.models['Ensemble'] = voting_clf
            model_scores['Ensemble'] = auc_ensemble
            print(f"✅ Ensemble AUC: {auc_ensemble:.3f}")

        best_model_name = max(model_scores, key=model_scores.get)
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name

        print(f"\n🏆 最佳模型: {best_model_name} (AUC: {model_scores[best_model_name]:.3f})")

        # ✅ 保存模型与特征列名
        joblib.dump(self.best_model, "risk_model.joblib")
        joblib.dump(self.feature_names, "model_feature_names.joblib")
        print("✅ 已保存模型到 risk_model.joblib")
        print("✅ 已保存特征列到 model_feature_names.joblib")

        # SHAP 分析器（仅限 XGBoost）
        if self.best_model_name == 'XGBoost':
            try:
                X_train_float = self.X_train.astype(np.float64)
                X_test_float = self.X_test.astype(np.float64)
                explainer = shap.Explainer(self.best_model, X_train_float)
                self.shap_values = explainer(X_test_float)
            except Exception as e:
                print(f"❌ SHAP 初始化失败: {str(e)}")
                self.shap_values = None
        else:
            print("⚠️ 当前最佳模型不是 XGBoost，跳过 SHAP 分析")
            self.shap_values = None

        return self.models
    
    def evaluate_model(self, model_name=None, custom_threshold=0.5):
        """
        6. 模型评估
        """
        print("\n" + "=" * 50)
        print("📈 模型评估")
        print("=" * 50)
        
        from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
        
        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model = self.models.get(model_name)
            if model is None:
                print(f"❌ 模型 {model_name} 不存在")
                return None
        
        # 预测
        y_proba = model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_proba >= custom_threshold).astype(int)
        
        # 评估指标
        auc_score = roc_auc_score(self.y_test, y_proba)
        
        print(f"模型: {model_name}")
        print(f"阈值: {custom_threshold}")
        print(f"AUC Score: {auc_score:.3f}")
        print("\n分类报告:")
        print(classification_report(self.y_test, y_pred))
        print("\n混淆矩阵:")
        print(confusion_matrix(self.y_test, y_pred))
        
        return {
            'auc_score': auc_score,
            'y_proba': y_proba,
            'y_pred': y_pred
        }
    
    def plot_threshold_analysis(self):
        """
        7. 阈值分析可视化
        """
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import precision_recall_curve
        
        if not hasattr(self, 'best_model'):
            print("❌ 请先训练模型")
            return
            
        y_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_proba)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(thresholds, precision[:-1], label="Precision", linewidth=2)
        plt.plot(thresholds, recall[:-1], label="Recall", linewidth=2)
        plt.plot(thresholds, f1[:-1], label="F1 Score", linewidth=2)
        plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Default (0.5)')
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Precision / Recall / F1 vs Threshold")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # ROC曲线
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        auc = roc_auc_score(self.y_test, y_proba)
        
        plt.subplot(2, 2, 2)
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 预测概率分布
        plt.subplot(2, 2, 3)
        plt.hist(y_proba[self.y_test == 0], bins=50, alpha=0.7, label='Non-risky', density=True)
        plt.hist(y_proba[self.y_test == 1], bins=50, alpha=0.7, label='Risky', density=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 特征重要性 (如果是树模型)
        plt.subplot(2, 2, 4)
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[-15:]  # top 15
            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Feature Importances')
        else:
            plt.text(0.5, 0.5, 'Feature importance\nnot available\nfor this model', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Feature Importance')
        
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self, top_n=20):
        """
        8. 获取特征重要性
        """
        if not hasattr(self, 'best_model'):
            print("❌ 请先训练模型")
            return None
            
        model = self.best_model
        
        # 不同模型的特征重要性获取方式
        if hasattr(model, 'feature_importances_'):
            # 树模型
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # 线性模型
            importances = np.abs(model.coef_[0])
        elif hasattr(model, 'estimators_'):
            # 集成模型 - 尝试获取平均重要性
            try:
                all_importances = []
                for name, estimator in model.named_estimators_.items():
                    if hasattr(estimator, 'feature_importances_'):
                        all_importances.append(estimator.feature_importances_)
                    elif hasattr(estimator, 'coef_'):
                        all_importances.append(np.abs(estimator.coef_[0]))
                
                if all_importances:
                    importances = np.mean(all_importances, axis=0)
                else:
                    print("⚠️ 无法获取集成模型的特征重要性")
                    return None
            except:
                print("⚠️ 无法获取集成模型的特征重要性")
                return None
        else:
            print("⚠️ 当前模型不支持特征重要性分析")
            return None
        
        # 创建特征重要性DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        print(f"\n🔍 Top {top_n} 重要特征 ({self.best_model_name}):")
        print(feature_importance_df.to_string(index=False))
        
        return feature_importance_df
    
    def generate_shap_summary(self, top_n=20):
        """
        9. 生成 SHAP 总结图
        """
        print("\n" + "=" * 50)
        print("🔍 SHAP 解释分析")
        print("=" * 50)

        if not hasattr(self, 'shap_values') or self.shap_values is None:
            print("❌ SHAP 解释尚未生成或不适用于当前模型")
            return

        # 画 SHAP 特征重要性图（类似条形图）
        shap.summary_plot(self.shap_values, self.X_test, plot_type='bar', max_display=top_n)

        # 画 SHAP 分布图（每个特征如何影响预测）
        shap.summary_plot(self.shap_values, self.X_test, max_display=top_n)

    def predict_and_explain(self, company_dict):
        """
        输入公司信息，输出风险预测结果，并用 SHAP force plot 可视化解释
        """
        import pandas as pd
        import numpy as np
        import shap
        import matplotlib.pyplot as plt

        if self.best_model is None or self.shap_values is None:
            print("❌ 请先训练模型并完成 SHAP 初始化")
            return

        # 1. 构造输入数据（复用前面函数逻辑）
        new_df = pd.DataFrame([company_dict])
        new_df['is_risky'] = 0  # 占位
        new_df['founded_year'] = pd.to_datetime(new_df['founded_at'], errors='coerce').dt.year

        if 'funding_total_usd' in new_df.columns:
            new_df['funding_total_usd'] = (
                new_df['funding_total_usd']
                .astype(str)
                .str.replace(r'[\$,]', '', regex=True)
                .replace(['', '-', 'nan'], np.nan)
                .astype(float)
            )
            if np.isnan(new_df['funding_total_usd'].iloc[0]):
                new_df['funding_total_usd'] = self.df_clean['funding_total_usd'].median()

        new_df['main_category'] = new_df['category_list'].apply(
            lambda x: str(x).split('|')[0] if '|' in str(x) else str(x)
        )
        top_countries = self.df_clean['country_code'].value_counts().head(10).index
        new_df['country_grouped'] = new_df['country_code'].apply(
            lambda x: x if x in top_countries else 'Other'
        )

        # One-hot 编码
        categorical_features = ['country_grouped', 'main_category']
        encoded = pd.get_dummies(new_df[categorical_features], prefix=['country', 'category'])

        for col in self.df_final.columns:
            if col.startswith("country_") or col.startswith("category_"):
                if col not in encoded.columns:
                    encoded[col] = 0

        numeric_features = ['founded_year', 'funding_total_usd', 'funding_rounds']
        available_numeric = [col for col in numeric_features if col in new_df.columns]

        final_input = pd.concat([
            new_df[available_numeric],
            encoded
        ], axis=1)

        final_input = final_input.reindex(columns=self.feature_names, fill_value=0)
        final_input = final_input.astype(np.float64)

        # 2. 做预测
        prob = self.best_model.predict_proba(final_input)[0][1]
        label = int(prob >= 0.5)

        print(f"\n🧪 预测结果: {'高风险 🚨' if label else '低风险 ✅'}，概率 = {prob:.3f}")

        # 3. 生成 SHAP force plot（解释为什么）
        explainer = shap.Explainer(self.best_model, self.X_train.astype(np.float64))
        shap_value = explainer(final_input)

        shap.plots.force(shap_value[0], matplotlib=True)
        plt.title("SHAP Force Plot - 风险预测解释")
        plt.show()

        return {'label': label, 'probability': prob}



def main():
    """
    主函数 - 完整的建模流程
    """
    # 初始化预测器
    predictor = StartupRiskPredictor("../dataset/big_startup_secsees_dataset.csv")
    
    # 执行完整流程
    try:
        # 1. 加载数据
        predictor.load_and_explore_data()
        
        # 2. 清洗数据
        predictor.clean_data()
        
        # 3. 特征工程
        predictor.feature_engineering()
        
        # 4. 划分数据集
        predictor.prepare_train_test_split()
        
        # 5. 训练模型
        predictor.train_models()
        
        # 6. 评估模型
        predictor.evaluate_model()
        
        # 7. 自定义阈值评估
        print("\n" + "="*50)
        print("🎯 自定义阈值分析")
        print("="*50)
        
        for threshold in [0.3, 0.4, 0.5]:
            print(f"\n--- 阈值 {threshold} ---")
            predictor.evaluate_model(custom_threshold=threshold)
        
        # 8. 可视化分析
        predictor.plot_threshold_analysis()
        
        # 9. 特征重要性
        predictor.get_feature_importance()

        # 10.shap分析
        predictor.generate_shap_summary()

        test_company_a = {
            'founded_at': '2010-05-20',                      # 成立较早
            'category_list': 'Biotechnology|Healthcare',     # 生物医药类（模型中表现较好）
            'country_code': 'USA',
            'funding_total_usd': '20000000',                 # 融资金额大
            'funding_rounds': 5                              # 融资轮数多
        }

        test_company_b = {
            'founded_at': '2018-11-01',                      # 成立时间较近
            'category_list': 'Games|Curated Web',            # 娱乐类（模型中风险高）
            'country_code': 'RUS',                           # 不在 Top10 国家，会被归为 "Other"
            'funding_total_usd': '30000',                    # 融资金额极低
            'funding_rounds': 0                              # 无融资历史
        }


        print("🎯 测试用例 A（低风险公司）")
        predictor.predict_and_explain(test_company_a)

        print("\n🎯 测试用例 B（高风险公司）")
        predictor.predict_and_explain(test_company_b)
        
        print("\n✅ 建模流程完成!")
        
    except Exception as e:
        print(f"❌ 建模过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()