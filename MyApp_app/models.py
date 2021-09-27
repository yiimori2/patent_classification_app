from django.db import models
from django.utils.timezone import now
from django_mysql.models import ListCharField

class ModelFile(models.Model):
    text = models.TextField(
        verbose_name = '発明内容',
        max_length = 500,
        )
    id = models.AutoField(primary_key=True)
    # 各特許分類を格納したリスト
    pred = ListCharField(
        base_field = models.CharField(max_length=20),
        size = 10,
        max_length = (10 * 21), # 10 * 20 character nominals, plus commas
        )
    # 各特許分類の信頼度を格納したリスト
    proba = ListCharField(
        base_field = models.IntegerField(blank=True, null=True),
        size = 10,
        max_length = (3 * 11),
        )
    input_date = models.DateField(default=now)

    # # 管理画面に表示方法を定義：必須項目が入っているかどうかで表示内容を分ける
    # # %s:文字列,%d:数値
    def __str__(self):
        if self.proba == None:
            return '%s, %d' % (self.input_date.strftime('%Y-%m-%d'), self.id)
        else:
            return '%s, %d, %s, %s' % (self.input_date.strftime('%Y-%m-%d'), self.id, self.pred, self.proba)